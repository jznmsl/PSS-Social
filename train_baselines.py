#!/usr/bin/env python3
"""
train_baselines.py - SARL and LSTM-RL Baselines 

Implements two additional baselines within the existing SB3 pipeline:

  1. SARL (Socially Attentive RL) — Chen et al. 2019
     Custom feature extractor with self-attention over K-NN neighbor embeddings.
     Uses standard PPO with a custom policy architecture.

  2. LSTM-RL — RecurrentPPO with MlpLstmPolicy from sb3-contrib.
     Tests whether temporal memory helps density generalization.

Both share the SAME observation space (115-dim), training protocol
(density randomization, VecNormalize), and evaluation pipeline.

Usage:
  # Train SARL baseline
  python train_baselines.py --agent SARL --seeds 42,123,456 --timesteps 10000000

  # Train LSTM-RL baseline
  python train_baselines.py --agent LSTM_RL --seeds 42,123,456 --timesteps 10000000

  # Train both
  python train_baselines.py --agent SARL,LSTM_RL --seeds 42,123,456

  # Quick test
  python train_baselines.py --agent SARL --seeds 42 --timesteps 100000 --n-envs 4
"""

from __future__ import annotations

__version__ = "1.1"

import os
import sys
import json
import time
import math
import argparse
import multiprocessing as mp
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces

# Local imports
from env_social_nav import (
    make_social_nav_env, SocialNavConfig,
    FIXED_OBS_DIM, MAX_NPCS, SCALAR_INV_DIM,
)

# For LSTM-RL
try:
    from sb3_contrib import RecurrentPPO
    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    SB3_CONTRIB_AVAILABLE = False
    print("WARNING: sb3-contrib not installed. LSTM_RL baseline unavailable.")
    print("  Install with: pip install sb3-contrib")


# ==============================================================================
# Observation Layout Constants (must match env_social_nav.py)
# ==============================================================================
EGO_DIM = 7          # vel(2) + pos(2) + goal_dir(2) + goal_dist(1)
K_NEIGHBORS = MAX_NPCS  # 24 — obs has 24 neighbor slots (for parsing)
K_ACTIVE_MAX = 16     # env's K_OBS_CAP — only first 16 slots ever have real data
NEIGHBOR_FEAT = 4     # rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y
NEIGHBOR_DIM = K_NEIGHBORS * NEIGHBOR_FEAT  # 96
SCALAR_DIM = SCALAR_INV_DIM  # 12
# Total: 7 + 96 + 12 = 115


# ==============================================================================
# SARL Feature Extractor
# ==============================================================================

class SARLFeatureExtractor(BaseFeaturesExtractor):
    """
    SARL feature extractor for SB3 (Chen et al., ICRA 2019).

    Faithfully implements the original SARL attention mechanism:
      1. Pairwise encoding: e_i = MLP(ego_state, neighbor_i) — joint input
      2. Mean embedding:    e_m = mean(e_i for active neighbors)
      3. Attention scores:  alpha_i = softmax(e_m^T W_a e_i)
      4. Crowd repr:        c = sum(alpha_i * e_i)
      5. Fusion:            output = MLP(ego_repr, c, scalar_repr)

    Key implementation details for padded variable-density observations:

      Soft gating (audit fix C3): The padding mask is produced by a learned
      gate network g_i = sigmoid(MLP(neighbor_i)). This is fully differentiable
      unlike boolean key_padding_mask, so the gate network receives gradients
      from the PPO loss throughout training.

      Hard mask for permanently-empty slots (audit fix M2): The env caps
      neighbor observations at K_ACTIVE_MAX=16 (env_social_nav.K_OBS_CAP).
      Slots 16-23 are always padding regardless of crowd density, so their
      gate is clamped to zero to save capacity.

      Pairwise encoding (audit fix M1): The original SARL embeds each neighbor
      jointly with the ego state, not neighbor features alone. This lets the
      encoder learn pairwise interaction patterns such as relative heading and
      approach geometry.

    Architecture:
      pairwise_encoder: Linear(11, 64) -> ReLU -> Linear(64, 64)
      gate_net:         Linear(4, 16) -> ReLU -> Linear(16, 1)
      W_a:              Linear(64, 64, bias=False)  — attention projection
      ego_encoder:      Linear(7, 64) -> ReLU
      scalar_encoder:   Linear(12, 32) -> ReLU
      fusion:           Linear(64+64+32, 128) -> ReLU -> Linear(128, 64) -> ReLU

    Output: 64-dim feature vector fed to PPO's policy and value heads.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        embed_dim: int = 64,
        num_heads: int = 4,       # kept in signature for compat; unused now
        features_dim: int = 64,
    ):
        super().__init__(observation_space, features_dim)

        self.embed_dim = embed_dim

        # Pairwise encoder: takes joint (ego, neighbor) = 7 + 4 = 11 dims
        self.pairwise_encoder = nn.Sequential(
            nn.Linear(EGO_DIM + NEIGHBOR_FEAT, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Attention projection matrix W_a (Chen et al. Eq. 5)
        # Computes: score_i = e_m^T @ W_a @ e_i
        self.W_a = nn.Linear(embed_dim, embed_dim, bias=False)

        # Soft gate network: predicts whether each slot holds a real neighbor.
        # Fully differentiable via sigmoid (audit fix C3 — replaces boolean mask).
        self.gate_net = nn.Sequential(
            nn.Linear(NEIGHBOR_FEAT, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # Register hard mask buffer for permanently-empty slots 16-23.
        # Shape: (1, K_NEIGHBORS) — broadcastable over batch dimension.
        hard_mask = torch.ones(1, K_NEIGHBORS)
        hard_mask[:, K_ACTIVE_MAX:] = 0.0
        self.register_buffer("hard_mask", hard_mask)

        # Ego state encoder (separate from pairwise — used in fusion)
        self.ego_encoder = nn.Sequential(
            nn.Linear(EGO_DIM, embed_dim),
            nn.ReLU(),
        )

        # Scalar summary encoder
        self.scalar_encoder = nn.Sequential(
            nn.Linear(SCALAR_DIM, 32),
            nn.ReLU(),
        )

        # Final fusion MLP
        fusion_dim = embed_dim + embed_dim + 32  # ego + crowd + scalars
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        # ── Split observation into components ──
        ego = observations[:, :EGO_DIM]                                       # (B, 7)
        neighbors_flat = observations[:, EGO_DIM:EGO_DIM + NEIGHBOR_DIM]      # (B, 96)
        scalars = observations[:, EGO_DIM + NEIGHBOR_DIM:]                    # (B, 12)

        # Reshape neighbors: (B, K, 4) where K=24
        neighbors = neighbors_flat.view(batch_size, K_NEIGHBORS, NEIGHBOR_FEAT)

        # ── Soft gating (C3 fix: differentiable, replaces boolean mask) ──
        # gate_i ∈ (0, 1): 1 = real neighbor, 0 = padding
        gate = torch.sigmoid(self.gate_net(neighbors)).squeeze(-1)     # (B, K)
        # Hard-zero slots 16-23 which are always padding (M2 fix)
        gate = gate * self.hard_mask                                   # (B, K)
        # Clamp total active count to avoid division by zero
        gate_sum = gate.sum(dim=1, keepdim=True).clamp(min=1.0)       # (B, 1)

        # ── Pairwise encoding (M1 fix: joint ego+neighbor input) ──
        # Expand ego to match neighbor dimension: (B, 7) -> (B, K, 7)
        ego_expanded = ego.unsqueeze(1).expand(-1, K_NEIGHBORS, -1)
        joint_input = torch.cat([ego_expanded, neighbors], dim=-1)     # (B, K, 11)
        embeds = self.pairwise_encoder(joint_input)                    # (B, K, 64)

        # ── Mean embedding (gate-weighted) ──
        # Weight each embedding by its gate value before averaging
        weighted_embeds = embeds * gate.unsqueeze(-1)                  # (B, K, 64)
        mean_embed = weighted_embeds.sum(dim=1) / gate_sum             # (B, 64)

        # ── Attention scores (Chen et al. Eq. 5) ──
        # score_i = e_m^T @ W_a @ e_i
        projected = self.W_a(embeds)                                   # (B, K, 64)
        # (B, 1, 64) @ (B, 64, K) -> (B, 1, K) -> (B, K)
        scores = torch.bmm(
            mean_embed.unsqueeze(1), projected.transpose(1, 2)
        ).squeeze(1)                                                   # (B, K)

        # Mask padding slots before softmax:
        # Set scores for padding (gate ≈ 0) to large negative so softmax → 0
        mask_penalty = (1.0 - gate) * (-1e9)
        scores = scores + mask_penalty
        alphas = torch.softmax(scores, dim=-1)                         # (B, K)

        # ── Crowd representation (gate-weighted attention pooling) ──
        # c = sum(alpha_i * e_i)
        crowd_repr = (alphas.unsqueeze(-1) * embeds).sum(dim=1)        # (B, 64)

        # ── Encode ego and scalars for fusion ──
        ego_repr = self.ego_encoder(ego)             # (B, 64)
        scalar_repr = self.scalar_encoder(scalars)   # (B, 32)

        # ── Fuse all representations ──
        fused = torch.cat([ego_repr, crowd_repr, scalar_repr], dim=1)  # (B, 160)
        return self.fusion(fused)


# ==============================================================================
# Scenario Mixing (ported from run_social.py)
# ==============================================================================

def parse_scenario_mix(mix_str: str, n_envs: int) -> List[str]:
    """
    Parse a scenario mix string and allocate envs proportionally.

    Example: "random:0.5,circle:0.5" with n_envs=64
             -> 32 random envs + 32 circle envs

    Returns a list of scenario names, one per env.
    """
    if not mix_str:
        return []

    parts = [p.strip() for p in mix_str.split(",")]
    scenarios = {}
    for part in parts:
        if ":" not in part:
            raise ValueError(f"Invalid scenario-mix format: '{part}'. Use 'scenario:fraction'")
        name, frac = part.split(":")
        scenarios[name.strip()] = float(frac.strip())

    total = sum(scenarios.values())
    if abs(total - 1.0) > 0.05:
        print(f"  WARNING: scenario-mix fractions sum to {total:.2f}, normalizing to 1.0")
        scenarios = {k: v / total for k, v in scenarios.items()}

    # Allocate envs proportionally
    allocation = []
    remaining = n_envs
    for i, (name, frac) in enumerate(scenarios.items()):
        if i == len(scenarios) - 1:
            # Last scenario gets all remaining envs (avoids rounding errors)
            count = remaining
        else:
            count = max(1, round(frac * n_envs))
            remaining -= count
        allocation.extend([name] * count)

    return allocation


# ==============================================================================
# Environment Factory (reused from run_social.py pattern)
# ==============================================================================

def make_env_fn(
    num_npcs: int,
    scenario: str,
    max_cycles: int,
    seed: int,
    randomize_density: bool = False,
    min_active_npcs: int = 10,
    max_active_npcs: int = 15,
):
    """Factory function to create social navigation environment."""
    def _init():
        social_config = SocialNavConfig()
        social_config.num_npcs = num_npcs
        social_config.scenario = scenario
        social_config.max_cycles = max_cycles
        social_config.randomize_density = randomize_density
        social_config.min_active_npcs = min_active_npcs
        social_config.max_active_npcs = max_active_npcs

        env = make_social_nav_env(
            num_npcs=num_npcs,
            scenario=scenario,
            max_cycles=max_cycles,
            config=social_config,
            randomize_density=randomize_density,
            min_active_npcs=min_active_npcs,
            max_active_npcs=max_active_npcs,
        )
        return env
    return _init


# ==============================================================================
# Training Callback (simplified from run_social.py)
# ==============================================================================

class BaselineCallback(BaseCallback):
    """Training callback with progress logging, early stopping, and best-model saving.

    Matches the log format of run_social.py's SocialCallback so training logs
    are consistent across all agents (SARL, LSTM_RL, Baseline, FIR variants).
    """

    def __init__(
        self,
        total_timesteps: int,
        agent_name: str = "",
        seed: int = 0,
        run_dir: str = "",
        early_stop: bool = True,
        early_stop_min_steps: int = 500_000,
        early_stop_patience: int = 5,
        print_interval_pct: int = 2,
        checkpoint_interval: int = 500_000,
    ):
        super().__init__()
        self.total = total_timesteps
        self.agent_name = agent_name
        self.seed = seed
        self.run_dir = run_dir
        self.early_stop = early_stop
        self.early_stop_min_steps = early_stop_min_steps
        self.early_stop_patience = early_stop_patience
        self.print_interval = print_interval_pct
        self.checkpoint_interval = checkpoint_interval
        self.last_checkpoint_step = 0

        self.prefix = f"[{agent_name:<16s} s{seed}]"
        self.start = None
        self.last_pct = -1
        self.episodes_completed = 0
        self.early_stopped = False
        self.consecutive_success_checks = 0

        # Episode-level metric buffers
        self.ep_goal_reached = []
        self.ep_safe_success = []
        self.ep_collisions = []
        self.ep_freezing_rates = []

        # Step-level metric buffers
        self.step_r_ext = []

        self.best_safe_success = -1.0
        self.best_model_step = 0

    def _on_training_start(self):
        self.start = time.time()

    def _on_step(self) -> bool:
        # Collect step-level reward (extrinsic — no FIR wrapper for baselines)
        rewards = self.locals.get("rewards", [])
        for r in rewards:
            self.step_r_ext.append(float(r))

        # Collect episode info from vectorized envs
        # CRITICAL: Use dones[i] to detect episode completion, NOT info dict keys.
        # The env returns (obs, reward, terminated, truncated, info) but SB3 VecEnv
        # merges terminated|truncated into a single done flag. The info dict does NOT
        # reliably contain "episode", "terminated", or "TimeLimit.truncated" for
        # goal-reached episodes — only dones[i] is authoritative.
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for i, info in enumerate(infos):
            if not isinstance(info, dict):
                continue

            done_i = dones[i] if isinstance(dones, (list, np.ndarray)) else dones
            if done_i:
                self.episodes_completed += 1
                cols = info.get("episode_collisions", 0)
                goal = info.get("goal_reached", False)
                frz = 1.0 if info.get("freezing", False) else 0.0

                self.ep_goal_reached.append(1.0 if goal else 0.0)
                self.ep_safe_success.append(1.0 if (goal and cols == 0) else 0.0)
                self.ep_collisions.append(cols)
                self.ep_freezing_rates.append(frz)

        # Trim buffers
        max_buf = 500
        for buf in [self.ep_goal_reached, self.ep_safe_success, self.ep_collisions, self.ep_freezing_rates]:
            if len(buf) > max_buf * 2:
                del buf[:len(buf) - max_buf]
        if len(self.step_r_ext) > max_buf * 2:
            del self.step_r_ext[:len(self.step_r_ext) - max_buf]

        # Console progress
        pct = int(self.num_timesteps / max(1, self.total) * 100)
        if pct >= self.last_pct + self.print_interval:
            self.last_pct = pct
            elapsed = time.time() - self.start
            fps = self.num_timesteps / max(1e-6, elapsed)
            remaining = self.total - self.num_timesteps
            eta = remaining / max(1, fps)

            recent_goal = np.mean(self.ep_goal_reached[-20:]) if self.ep_goal_reached else 0
            recent_safe = np.mean(self.ep_safe_success[-20:]) if self.ep_safe_success else 0
            recent_col = np.mean(self.ep_collisions[-20:]) if self.ep_collisions else 0
            recent_frz = np.mean(self.ep_freezing_rates[-20:]) if self.ep_freezing_rates else 0
            recent_r_ext = np.mean(self.step_r_ext[-100:]) if self.step_r_ext else 0

            def _fmt_time(secs):
                if secs > 3600:
                    return f"{int(secs//3600)}h {int(secs%3600//60):02d}m {int(secs%60):02d}s"
                return f"{int(secs//60)}m {int(secs%60):02d}s"

            print(
                f"  {self.prefix} {pct:3d}% | {self.num_timesteps:>9,} | "
                f"T:{_fmt_time(elapsed)} | FPS:{fps:,.0f} | ETA:{_fmt_time(eta)} | "
                f"Ep:{self.episodes_completed:,} | "
                f"Goal:{recent_goal:.0%} | Safe:{recent_safe:.0%} | Col:{recent_col:.2f} | "
                f"r_ext:{recent_r_ext:+.3f} | Frz:{recent_frz:.0%}",
                flush=True,
            )

            # Save best model
            if self.run_dir and self.ep_safe_success:
                current_ss = float(np.mean(self.ep_safe_success[-200:]))
                if current_ss > self.best_safe_success and self.episodes_completed >= 20:
                    self.best_safe_success = current_ss
                    self.best_model_step = self.num_timesteps
                    try:
                        self.model.save(os.path.join(self.run_dir, "best_model"))
                        env = self.model.get_env()
                        if hasattr(env, "save"):
                            env.save(os.path.join(self.run_dir, "best_vecnorm.pkl"))
                        print(f"  {self.prefix} New best! SafeSuccess={current_ss:.1%} @ step {self.num_timesteps:,}", flush=True)
                    except Exception:
                        pass

            # Early stopping
            if self.early_stop and self.num_timesteps >= self.early_stop_min_steps:
                if recent_safe >= 1.0:
                    self.early_stopped = True
                    print(f"\n  {self.prefix} EARLY STOP: SafeSuccess 100%!")
                    return False
                elif recent_safe >= 0.98:
                    self.consecutive_success_checks += 1
                    if self.consecutive_success_checks >= self.early_stop_patience:
                        self.early_stopped = True
                        print(f"\n  {self.prefix} EARLY STOP: SafeSuccess {recent_safe:.0%} stable")
                        return False
                else:
                    self.consecutive_success_checks = 0

        # Periodic checkpoint (crash insurance)
        if self.run_dir and self.checkpoint_interval > 0:
            if self.num_timesteps - self.last_checkpoint_step >= self.checkpoint_interval:
                self.last_checkpoint_step = self.num_timesteps
                try:
                    ckpt_path = os.path.join(self.run_dir, f"checkpoint_{self.num_timesteps}")
                    self.model.save(ckpt_path)
                    env = self.model.get_env()
                    if hasattr(env, "save"):
                        env.save(os.path.join(self.run_dir, f"checkpoint_{self.num_timesteps}_vecnorm.pkl"))
                    print(f"  {self.prefix} Checkpoint @ step {self.num_timesteps:,}", flush=True)
                except Exception:
                    pass

        return True

    def _on_training_end(self):
        elapsed = time.time() - self.start
        if self.early_stopped:
            print(f"\n  {self.prefix} Training Early Stopped!", flush=True)
            steps_saved = self.total - self.num_timesteps
            print(f"     Steps completed: {self.num_timesteps:,} / {self.total:,}", flush=True)
            print(f"     Steps saved: {steps_saved:,} ({100*steps_saved/self.total:.0f}%)", flush=True)
        else:
            print(f"\n  {self.prefix} Training Complete!", flush=True)
        print(f"     Total time: {elapsed/60:.1f} min | Episodes: {self.episodes_completed}")
        if self.ep_safe_success:
            print(f"     Final SafeSuccess: {np.mean(self.ep_safe_success[-100:]):.1%}")
        if self.ep_collisions:
            print(f"     Final Col/Ep: {np.mean(self.ep_collisions[-100:]):.2f}")
        if self.best_model_step > 0:
            print(f"     Best SafeSuccess: {self.best_safe_success:.1%} @ step {self.best_model_step:,}")


# ==============================================================================
# Model Builders
# ==============================================================================

def build_sarl_model(
    venv,
    seed: int,
    n_steps: int = 2048,
    batch_size: int = 128,
    n_epochs: int = 10,
    device: str = "cpu",
    tb_log_dir: str = None,
) -> PPO:
    """Build PPO with SARL feature extractor."""
    policy_kwargs = dict(
        features_extractor_class=SARLFeatureExtractor,
        features_extractor_kwargs=dict(
            embed_dim=64,
            num_heads=4,
            features_dim=64,
        ),
        net_arch=dict(pi=[64], vf=[64]),  # Smaller heads since feature extractor is richer
    )

    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=policy_kwargs,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=3e-4,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=device,
        seed=seed,
        tensorboard_log=tb_log_dir,
    )
    return model


def build_lstm_model(
    venv,
    seed: int,
    n_steps: int = 2048,
    batch_size: int = 128,
    n_epochs: int = 10,
    device: str = "cpu",
    tb_log_dir: str = None,
) -> "RecurrentPPO":
    """Build RecurrentPPO with LSTM policy.

    CRITICAL (audit C1): RecurrentPPO processes rollouts as intact
    per-environment sequences. The number of minibatches must divide
    n_envs (not the total rollout). With n_envs=16, valid n_minibatches
    are {1, 2, 4, 8, 16}, giving batch_size = n_envs * n_steps / n_mb.
    The requested batch_size is adjusted upward to the nearest valid value.
    """
    if not SB3_CONTRIB_AVAILABLE:
        raise ImportError(
            "sb3-contrib is required for LSTM_RL. "
            "Install with: pip install sb3-contrib"
        )

    # Compute valid batch_size for RecurrentPPO
    n_envs = venv.num_envs
    rollout = n_envs * n_steps
    # Find largest n_minibatches such that:
    #   (a) n_minibatches divides n_envs (sequence integrity)
    #   (b) resulting batch_size >= requested (don't make batches too huge)
    # Fallback: single minibatch (batch_size = full rollout)
    valid_batch_size = rollout  # worst case: 1 minibatch
    for n_mb in sorted([i for i in range(1, n_envs + 1) if n_envs % i == 0], reverse=True):
        candidate = rollout // n_mb
        if candidate >= batch_size:
            valid_batch_size = candidate
            break

    if valid_batch_size != batch_size:
        print(f"  [LSTM_RL] Adjusted batch_size {batch_size} -> {valid_batch_size} "
              f"(RecurrentPPO requires n_envs={n_envs} divisible by n_minibatches={rollout // valid_batch_size})")
    batch_size = valid_batch_size

    model = RecurrentPPO(
        "MlpLstmPolicy",
        venv,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=3e-4,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=device,
        seed=seed,
        tensorboard_log=tb_log_dir,
    )
    return model


# ==============================================================================
# Core Training Function
# ==============================================================================

def train_one(
    agent_name: str,
    seed: int,
    out_dir: str,
    num_npcs: int,
    scenario: str,
    max_cycles: int,
    timesteps: int,
    n_envs: int,
    vecenv_kind: str,
    device: str,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    randomize_density: bool,
    min_active_npcs: int,
    max_active_npcs: int,
    early_stop: bool = True,
    scenario_mix: str = "",
) -> Tuple[str, int, str, float]:
    """
    Train one agent+seed combination.
    Returns: (agent_name, seed, status, elapsed_time)
    """
    run_start = time.time()

    try:
        set_random_seed(seed)

        run_dir = os.path.join(out_dir, agent_name, f"seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)

        # Build vectorized environment — with scenario mixing support
        # When scenario_mix is set (e.g. "random:0.5,circle:0.5"),
        # different envs in the VecEnv are assigned different scenarios.
        scenario_allocation = parse_scenario_mix(scenario_mix, n_envs) if scenario_mix else []

        if scenario_allocation:
            from collections import Counter
            mix_counts = Counter(scenario_allocation)
            mix_display = ", ".join(f"{s}={c}" for s, c in mix_counts.items())
            env_fns = [
                make_env_fn(
                    num_npcs, scenario_allocation[i], max_cycles, seed + 1000 * i,
                    randomize_density=randomize_density,
                    min_active_npcs=min_active_npcs,
                    max_active_npcs=max_active_npcs,
                )
                for i in range(n_envs)
            ]
        else:
            mix_display = ""
            env_fns = [
                make_env_fn(
                    num_npcs, scenario, max_cycles, seed + 1000 * i,
                    randomize_density=randomize_density,
                    min_active_npcs=min_active_npcs,
                    max_active_npcs=max_active_npcs,
                )
                for i in range(n_envs)
            ]

        if vecenv_kind == "subproc" and n_envs > 1:
            venv = SubprocVecEnv(env_fns, start_method="spawn")
        else:
            venv = DummyVecEnv(env_fns)

        # VecNormalize (same as Baseline)
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

        # Build model
        tb_log_dir = os.path.join(run_dir, "tb_logs")

        print(f"\n{'='*70}")
        print(f"  {agent_name} | seed={seed}")
        print(f"  num_npcs={num_npcs}, scenario={scenario}, max_cycles={max_cycles}")
        if mix_display:
            print(f"  SCENARIO MIX: {mix_display} (total {n_envs} envs)")
        print(f"  n_envs={n_envs}, vecenv={vecenv_kind}, device={device}")
        print(f"  timesteps={timesteps:,}")
        if randomize_density:
            print(f"  DENSITY RANDOMIZATION: {min_active_npcs}-{max_active_npcs} NPCs")
        print(f"{'='*70}")

        if agent_name == "SARL":
            model = build_sarl_model(
                venv, seed, n_steps, batch_size, n_epochs, device, tb_log_dir
            )
            print(f"  Architecture: PPO + SARL feature extractor (attention over K-NN)")
            # Print parameter count
            total_params = sum(p.numel() for p in model.policy.parameters())
            print(f"  Total parameters: {total_params:,}")

        elif agent_name == "LSTM_RL":
            model = build_lstm_model(
                venv, seed, n_steps, batch_size, n_epochs, device, tb_log_dir
            )
            print(f"  Architecture: RecurrentPPO + MlpLstmPolicy")
            total_params = sum(p.numel() for p in model.policy.parameters())
            print(f"  Total parameters: {total_params:,}")

        else:
            raise ValueError(f"Unknown agent: {agent_name}. Use SARL or LSTM_RL.")

        # Save metadata
        # Read actual batch_size from model (may differ from input for LSTM_RL)
        actual_batch_size = model.batch_size if hasattr(model, 'batch_size') else batch_size
        meta = {
            "agent_name": agent_name,
            "exp_name": agent_name,  # Compatibility with eval_unified.py
            "seed": seed,
            "num_npcs": num_npcs,
            "scenario": scenario,
            "max_cycles": max_cycles,
            "timesteps": timesteps,
            "n_envs": n_envs,
            "vecenv_kind": vecenv_kind,
            "device": device,
            "ppo": {
                "n_steps": n_steps,
                "batch_size": actual_batch_size,
                "n_epochs": n_epochs,
            },
            "density_randomization": {
                "enabled": randomize_density,
                "min_active_npcs": min_active_npcs,
                "max_active_npcs": max_active_npcs,
            },
            "scenario_mix": scenario_mix if scenario_mix else "none",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Train
        callback = BaselineCallback(
            timesteps, agent_name, seed,
            run_dir=run_dir,
            early_stop=early_stop,
        )
        model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)

        # Save final model
        model.save(os.path.join(run_dir, "final_model"))
        venv.save(os.path.join(run_dir, "vecnorm.pkl"))
        venv.close()

        elapsed = time.time() - run_start
        return (agent_name, seed, "OK", elapsed)

    except Exception as e:
        elapsed = time.time() - run_start
        import traceback
        traceback.print_exc()
        return (agent_name, seed, f"FAIL: {str(e)[:80]}", elapsed)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train SARL and LSTM-RL baselines"
    )

    # Agent selection
    parser.add_argument(
        "--agent", type=str, default="SARL",
        help="Comma-separated list of agents to train: SARL, LSTM_RL (default: SARL)"
    )

    # Seeds
    parser.add_argument("--seeds", type=str, default="42,123,456")

    # Training
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--vecenv", type=str, default="subproc", choices=["dummy", "subproc"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=10)

    # Environment
    parser.add_argument("--num-npcs-train", type=int, default=15,
                        help="Fixed NPC count (when density randomization is off)")
    parser.add_argument("--scenario", type=str, default="random",
                        choices=["corridor", "intersection", "circle", "random"])
    parser.add_argument("--scenario-mix", type=str, default="",
                        help="Scenario mixing, e.g. 'random:0.5,circle:0.5'. "
                             "Overrides --scenario when set.")
    parser.add_argument("--max-cycles", type=int, default=100)

    # Density randomization (matching existing Baseline training protocol)
    parser.add_argument("--randomize-density", action="store_true", default=True,
                        help="Randomize NPC count each episode (default: True)")
    parser.add_argument("--no-randomize-density", action="store_true",
                        help="Disable density randomization")
    parser.add_argument("--min-active-npcs", type=int, default=10)
    parser.add_argument("--max-active-npcs", type=int, default=15)

    # Early stopping
    parser.add_argument("--no-early-stop", action="store_true")

    # Output
    parser.add_argument("--output", type=str, default="./runs_social",
                        help="Output directory (default: ./runs_social)")

    args = parser.parse_args()

    # Parse agents and seeds
    agents = [a.strip() for a in args.agent.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.no_randomize_density:
        args.randomize_density = False

    # Validate
    for agent in agents:
        if agent == "LSTM_RL" and not SB3_CONTRIB_AVAILABLE:
            print(f"ERROR: sb3-contrib required for LSTM_RL. Install: pip install sb3-contrib")
            sys.exit(1)
        if agent not in ("SARL", "LSTM_RL"):
            print(f"ERROR: Unknown agent '{agent}'. Use SARL or LSTM_RL.")
            sys.exit(1)

    # Ensure batch_size divides rollout
    rollout_size = args.n_envs * args.n_steps
    while rollout_size % args.batch_size != 0:
        args.batch_size //= 2

    print("=" * 80)
    print("Baseline Training")
    print("=" * 80)
    print(f"  Agents:     {agents}")
    print(f"  Seeds:      {seeds}")
    print(f"  Timesteps:  {args.timesteps:,}")
    print(f"  Env:        scenario={args.scenario}, num_npcs={args.num_npcs_train}")
    print(f"  Training:   n_envs={args.n_envs}, n_steps={args.n_steps}, batch={args.batch_size}")
    if args.randomize_density:
        print(f"  Density randomization: {args.min_active_npcs}-{args.max_active_npcs} NPCs")
    if args.scenario_mix:
        print(f"  Scenario mix: {args.scenario_mix}")
    print(f"  Output:     {args.output}")
    print("=" * 80)

    results = []
    t0 = time.time()

    for agent in agents:
        for seed in seeds:
            result = train_one(
                agent_name=agent,
                seed=seed,
                out_dir=args.output,
                num_npcs=args.num_npcs_train,
                scenario=args.scenario,
                max_cycles=args.max_cycles,
                timesteps=args.timesteps,
                n_envs=args.n_envs,
                vecenv_kind=args.vecenv,
                device=args.device,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                randomize_density=args.randomize_density,
                min_active_npcs=args.min_active_npcs,
                max_active_npcs=args.max_active_npcs,
                early_stop=not args.no_early_stop,
                scenario_mix=args.scenario_mix,
            )
            results.append(result)

    # Summary
    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print("ALL RUNS COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time/60:.1f} min")
    print()
    print(f"  {'Agent':<12} {'Seed':<8} {'Status':<12} {'Time':<12}")
    print(f"  {'-'*12} {'-'*8} {'-'*12} {'-'*12}")
    for agent, seed, status, elapsed in results:
        t_str = f"{elapsed/60:.1f}min"
        print(f"  {agent:<12} {seed:<8} {status:<12} {t_str:<12}")

    successful = sum(1 for r in results if r[2] == "OK")
    print(f"\n  Successful: {successful}/{len(results)}")

    print("\nNEXT STEPS:")
    print(f"  # Evaluate (density sweep):")
    print(f"  python eval_unified.py {args.output} --scenario random --densities 11,13,15,17,19,21,23")
    print(f"  python eval_unified.py {args.output} --scenario circle --densities 11,13,15,17,19,21,23")
    print("=" * 80)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()