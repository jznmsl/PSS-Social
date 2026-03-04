#!/usr/bin/env python3
"""
pss_social.py - Proxemic Shaping for Safe Social Navigation 

VERSION: 5.0 (Position-only Phi, velocity removed)

This module provides Potential-based Social Shaping (PSS) as intrinsic reward
for training DRL social navigation agents. PSS encodes proxemic social norms
(intimate space, personal space) as a potential function, yielding a
reward-shaping signal r = gamma * Phi(s') - Phi(s).

Key features:
  1. Position-only Phi: proxemic costs depend on distance only.
     Velocity was removed after ablation showed it teaches speed modulation
     rather than spatial avoidance, degrading OOD generalization.
  2. Density-adaptive scaling: normalizes aggregate Phi penalty by local
     NPC count, keeping the shaping signal density-invariant.
  3. Collision penalty: discrete signal for hard constraint learning.

Experiment configurations (paper models):
  - Baseline: PPO + collision penalty (no shaping)
  - PSS_Social: PPO + density-scaled PSS + collision penalty (our method)
  - PSS_Only_V0: PPO + raw PSS + collision penalty (ablation: no density)
  - PSS_Only_V1: PPO + density-scaled PSS + collision penalty (ablation)
"""

from __future__ import annotations

__version__ = "5.0-position-only"

import numpy as np

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


# ==============================================================================
# Utils
# ==============================================================================

def sanitize_array(x: np.ndarray, name: str = "array") -> np.ndarray:
    """Replace NaN/inf with 0 to prevent cascading numerical issues."""
    if not isinstance(x, np.ndarray):
        return x
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class PSSSocialConfig:
    """Configuration for PSS Social Navigation."""

    # Action dimension (MPE continuous_actions=True uses 5-dim)
    action_dim: int = 5

    # Fixed observation support (must match env_social_nav.py MAX_NPCS)
    max_npcs: int = 24

    # Social Force parameters (used by PSS intimate cost)
    k_rep: float = 5.0

    # PSS weight (beta3)
    beta3_init: float = 1.0
    beta3_final: float = 0.5
    beta3_anneal: bool = True
    anneal_start_ratio: float = 0.7

    max_training_steps: int = 1_000_000
    gamma: float = 0.99

    # Proxemic zone thresholds
    intimate_space: float = 0.3
    personal_space: float = 0.5

    # PSS weights
    pss_progress_w: float = 1.0
    pss_intimate_w: float = 3.0
    pss_personal_w: float = 0.5
    pss_reward_clip: float = 5.0

    # Density-adaptive scaling (normalizes Phi costs by local NPC count)
    pss_density_adaptive: bool = False

    # Collision penalty (discrete signal)
    collision_penalty: float = 0.0

    # Kept for backward compat with run_social.py config-based use_fir gate.
    # Always 0.0 in PSS-only version.
    beta1_init: float = 0.0
    beta2_target: float = 0.0


# ==============================================================================
# Experiment Configs
# ==============================================================================

def get_social_experiment_config(experiment_name: str) -> PSSSocialConfig:
    """Get configuration for social navigation experiments."""
    c = PSSSocialConfig()

    if experiment_name == "Baseline":
        # Plain PPO (no shaping). Collision penalty comes from env (-5.0).
        c.beta3_init = 0.0
        c.beta3_final = 0.0
        c.beta3_anneal = False
        c.collision_penalty = 0.0

    elif experiment_name == "Safe_Baseline":
        # PPO with high collision penalty (induces freezing)
        c.beta3_init = 0.0
        c.beta3_final = 0.0
        c.beta3_anneal = False
        c.collision_penalty = 10.0

    elif experiment_name == "PSS_Social":
        # Full method: density-scaled PSS + collision penalty + anneal
        c.beta3_init = 2.0
        c.beta3_final = 1.0
        c.beta3_anneal = True
        c.anneal_start_ratio = 0.6
        c.pss_progress_w = 1.5
        c.pss_intimate_w = 8.0
        c.pss_personal_w = 1.2
        c.pss_density_adaptive = True
        c.collision_penalty = 5.0

    elif experiment_name == "PSS_Social_Moderate":
        c.beta3_init = 2.0
        c.beta3_final = 1.0
        c.beta3_anneal = True
        c.anneal_start_ratio = 0.6
        c.pss_progress_w = 1.5
        c.pss_intimate_w = 4.0
        c.pss_personal_w = 0.8
        c.pss_density_adaptive = True
        c.collision_penalty = 5.0

    elif experiment_name == "PSS_Social_Safe":
        c.beta3_init = 2.0
        c.beta3_final = 1.0
        c.beta3_anneal = True
        c.anneal_start_ratio = 0.6
        c.intimate_space = 0.35
        c.personal_space = 0.6
        c.pss_progress_w = 1.5
        c.pss_intimate_w = 4.0
        c.pss_personal_w = 1.5
        c.pss_density_adaptive = True
        c.collision_penalty = 5.0

    # ==================================================================
    # PSS Ablation: V0 (raw Phi) vs V1 (density-scaled Phi)
    # ==================================================================

    elif experiment_name == "PSS_Only_V0":
        # PPO + PSS (position-only Phi, no density scaling)
        c.beta3_init = 2.0
        c.beta3_final = 2.0
        c.beta3_anneal = False
        c.pss_progress_w = 1.5
        c.pss_intimate_w = 4.0
        c.pss_personal_w = 0.8
        c.pss_density_adaptive = False
        c.collision_penalty = 0.0

    elif experiment_name == "PSS_Only_V1":
        # PPO + PSS + density scaling
        c.beta3_init = 2.0
        c.beta3_final = 2.0
        c.beta3_anneal = False
        c.pss_progress_w = 1.5
        c.pss_intimate_w = 4.0
        c.pss_personal_w = 0.8
        c.pss_density_adaptive = True
        c.collision_penalty = 0.0

    return c


# ==============================================================================
# PSS Module
# ==============================================================================

class SocialPSS_Module:
    """
    Potential-Based Social Shaping for social navigation.

    Phi(s) = progress - intimate_cost - personal_cost
    r_pss = gamma * Phi(s') - Phi(s)
    """

    def __init__(self, config: PSSSocialConfig):
        self.c = config
        self.last_phi: Dict[str, float] = {}
        self.agent_radius: float = 0.15

    def reset(self):
        self.last_phi.clear()

    def _intimate_cost(self, pos, neighbors_pos):
        """High penalty for being inside intimate space. Position-only."""
        if neighbors_pos.size == 0:
            return 0.0

        diff = neighbors_pos - pos
        dists = np.linalg.norm(diff, axis=1)

        mask = dists < self.c.intimate_space
        if not np.any(mask):
            return 0.0

        penetrations = self.c.intimate_space - dists[mask]
        costs = self.c.k_rep * np.exp(np.clip(penetrations / 0.1, -50, 50))

        return float(np.sum(costs))

    def _personal_cost(self, pos, neighbors_pos):
        """Soft penalty for personal space intrusion. Position-only."""
        if neighbors_pos.size == 0:
            return 0.0

        diff = neighbors_pos - pos
        dists = np.linalg.norm(diff, axis=1)

        mask = (dists >= self.c.intimate_space) & (dists < self.c.personal_space)
        if not np.any(mask):
            return 0.0

        penetrations = self.c.personal_space - dists[mask]
        costs = penetrations * 2.0

        return float(np.sum(costs))

    def _progress_term(self, pos, goal_pos):
        """Negative linear distance to goal (constant gradient)."""
        dist_to_goal = np.linalg.norm(pos - goal_pos)
        return float(-self.c.pss_progress_w * dist_to_goal)

    def get_phi(self, pos, neighbors_pos, goal_pos):
        """Compute potential Phi(s). Higher = better state. Position-only."""
        pos = sanitize_array(pos.astype(np.float32), "pss_pos")
        neighbors_pos = sanitize_array(neighbors_pos.astype(np.float32), "pss_neighbors")
        goal_pos = sanitize_array(goal_pos.astype(np.float32), "pss_goal")

        progress = self._progress_term(pos, goal_pos)
        intimate_cost = self.c.pss_intimate_w * self._intimate_cost(
            pos, neighbors_pos)
        personal_cost = self.c.pss_personal_w * self._personal_cost(
            pos, neighbors_pos)

        if self.c.pss_density_adaptive and neighbors_pos.size > 0:
            dists = np.linalg.norm(neighbors_pos - pos, axis=1)
            n_nearby = float(np.sum(dists < self.c.personal_space * 2.0))
            density_scale = 1.0 / max(1.0, n_nearby / 3.0)
            intimate_cost *= density_scale
            personal_cost *= density_scale

        phi = progress - intimate_cost - personal_cost
        return float(phi)

    def shaping_reward(self, agent_key, phi, done):
        """Compute shaping reward: r = gamma * Phi(s') - Phi(s)"""
        if done or agent_key not in self.last_phi:
            self.last_phi[agent_key] = float(phi)
            return 0.0

        r = float(self.c.gamma * phi - self.last_phi[agent_key])
        self.last_phi[agent_key] = float(phi)
        r = float(np.clip(r, -self.c.pss_reward_clip, self.c.pss_reward_clip))
        return r


# ==============================================================================
# Local Wrapper (per-environment)
# ==============================================================================

class PSSSocialLocalWrapper(gym.Wrapper):
    """Local wrapper that computes PSS data for a single environment."""

    def __init__(self, env, config):
        super().__init__(env)
        self.c = config
        self.pss = SocialPSS_Module(config)
        self.need_pss = (config.beta3_init > 0.0) or (config.beta3_final > 0.0)

        self.episode_collisions = 0
        self.episode_intrusions = 0
        self.episode_freezing_steps = 0
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.goal_reached = False
        self.goal_reached_step = -1

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.pss.reset()
        self.episode_collisions = 0
        self.episode_intrusions = 0
        self.episode_freezing_steps = 0
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.goal_reached = False
        self.goal_reached_step = -1
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)

        self.episode_steps += 1
        self.episode_reward += reward

        self.episode_collisions = info.get("episode_collisions", self.episode_collisions)
        self.episode_intrusions = info.get("episode_intrusions", self.episode_intrusions)
        if info.get("freezing", False):
            self.episode_freezing_steps += 1
        if info.get("goal_reached", False) and not self.goal_reached:
            self.goal_reached = True
            self.goal_reached_step = self.episode_steps

        # Get agent state for PSS
        ego_pos = np.zeros(2, dtype=np.float32)
        ego_vel = np.zeros(2, dtype=np.float32)
        goal_pos = np.zeros(2, dtype=np.float32)
        npc_pos = np.zeros((0, 2), dtype=np.float32)
        npc_vel = np.zeros((0, 2), dtype=np.float32)

        try:
            world = self.env.unwrapped.world
            ego = world.agents[0]
            ego_pos = ego.state.p_pos.copy()
            ego_vel = ego.state.p_vel.copy()
            goal_pos = getattr(ego, 'goal_pos', np.zeros(2))
            npc_pos = np.array([a.state.p_pos for a in world.agents[1:]], dtype=np.float32)
            npc_vel = np.array([a.state.p_vel for a in world.agents[1:]], dtype=np.float32)
        except Exception:
            pass

        info["ego_velocity"] = float(np.linalg.norm(ego_vel))

        r_pss = 0.0
        if self.need_pss:
            phi = self.pss.get_phi(ego_pos, npc_pos, goal_pos)
            r_pss = self.pss.shaping_reward("ego", phi, done)

        info["fir_data"] = {"r_pss": r_pss}

        if done:
            info["episode_collisions"] = self.episode_collisions
            info["episode_intrusions"] = self.episode_intrusions
            info["episode_freezing_steps"] = self.episode_freezing_steps
            info["episode_steps"] = self.episode_steps
            info["freezing_rate"] = self.episode_freezing_steps / max(1, self.episode_steps)
            info["goal_reached"] = self.goal_reached
            info["time_to_goal"] = self.goal_reached_step if self.goal_reached else -1
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.episode_steps,
                "t": self.episode_steps,
            }

        return obs, reward, terminated, truncated, info


# ==============================================================================
# Global Vec Wrapper
# ==============================================================================

class PSSSocialGlobalVecWrapper:
    """Wraps a VecEnv, adds PSS intrinsic rewards and collision penalty."""

    def __init__(self, venv, config, device="cpu"):
        self.venv = venv
        self.c = config
        self.device = device
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.total_steps = 0

    def close(self):
        return self.venv.close()

    def reset(self):
        self.total_steps = 0
        return self.venv.reset()

    def seed(self, seed=None):
        if hasattr(self.venv, 'seed'):
            return self.venv.seed(seed)
        return None

    def env_is_wrapped(self, wrapper_class, indices=None):
        if hasattr(self.venv, 'env_is_wrapped'):
            return self.venv.env_is_wrapped(wrapper_class, indices)
        return [False] * self.num_envs

    def env_method(self, method_name, *args, indices=None, **kwargs):
        if hasattr(self.venv, 'env_method'):
            return self.venv.env_method(method_name, *args, indices=indices, **kwargs)
        return None

    def get_attr(self, attr_name, indices=None):
        if hasattr(self.venv, 'get_attr'):
            return self.venv.get_attr(attr_name, indices)
        return None

    def set_attr(self, attr_name, value, indices=None):
        if hasattr(self.venv, 'set_attr'):
            return self.venv.set_attr(attr_name, value, indices)

    def step_async(self, actions):
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.total_steps += self.num_envs

        # Beta3 (PSS) schedule
        progress = min(1.0, self.total_steps / max(1, self.c.max_training_steps))
        beta3 = self.c.beta3_init
        if self.c.beta3_anneal and progress > self.c.anneal_start_ratio:
            rem = (progress - self.c.anneal_start_ratio) / max(1e-6, 1.0 - self.c.anneal_start_ratio)
            beta3 = self.c.beta3_init + rem * (self.c.beta3_final - self.c.beta3_init)

        # Add PSS intrinsic reward
        for i, info in enumerate(infos):
            fd = info.get("fir_data") if isinstance(info, dict) else None
            if fd is not None:
                r_pss = fd.get("r_pss", 0.0)
                r_int = beta3 * r_pss
                r_ext = float(rewards[i])
                rewards[i] = float(rewards[i] + r_int)
                infos[i]["beta3"] = beta3
                infos[i]["r_ext"] = r_ext
                infos[i]["r_int"] = r_int
                infos[i]["r_pss"] = r_pss

        # Apply collision penalty to ALL envs
        for i, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            collisions = info.get("collisions", 0)
            r_cost = self.c.collision_penalty * collisions
            if r_cost > 0:
                rewards[i] = float(rewards[i] - r_cost)
            infos[i]["r_cost"] = float(r_cost)

        return obs, rewards, dones, infos


# ==============================================================================
# Test
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing PSS Social Navigation Module (v5.0)")
    print("=" * 60)

    # ── Test 1: Config verification ──
    print("\n  Config table:")
    print(f"  {'Experiment':<20} {'beta3':>6} {'den':>5} {'col_pen':>8} {'intim_w':>8}")
    print(f"  {'-'*52}")
    for name in ["Baseline", "PSS_Social", "PSS_Only_V0", "PSS_Only_V1"]:
        c = get_social_experiment_config(name)
        print(f"  {name:<20} {c.beta3_init:6.1f} {str(c.pss_density_adaptive):>5} "
              f"{c.collision_penalty:8.1f} {c.pss_intimate_w:8.1f}")

    # ── Test 2: Collision penalty checks ──
    cb = get_social_experiment_config("Baseline")
    cs = get_social_experiment_config("PSS_Social")
    c0 = get_social_experiment_config("PSS_Only_V0")
    c1 = get_social_experiment_config("PSS_Only_V1")

    assert cb.collision_penalty == 5.0, f"Baseline cp={cb.collision_penalty}, want 5"
    assert cs.collision_penalty == 10.0, f"PSS_Social cp={cs.collision_penalty}, want 10"
    assert c0.collision_penalty == 5.0, f"V0 cp={c0.collision_penalty}, want 5"
    assert c1.collision_penalty == 5.0, f"V1 cp={c1.collision_penalty}, want 5"
    print("\n  Collision penalties: PASS (BL=5, PSS=10, V0=5, V1=5)")

    # ── Test 3: Velocity fully removed ──
    import inspect
    for method_name in ['_intimate_cost', '_personal_cost', 'get_phi']:
        sig = inspect.signature(getattr(SocialPSS_Module, method_name))
        params = list(sig.parameters.keys())
        assert 'ego_vel' not in params, f"{method_name} still has ego_vel!"
        assert 'neighbors_vel' not in params, f"{method_name} still has neighbors_vel!"
    assert not hasattr(cs, 'pss_velocity_aware'), "pss_velocity_aware still in config!"
    print("  Velocity removed: PASS (no velocity args anywhere)")

    # ── Test 4: Density scaling ──
    assert cs.pss_density_adaptive == True, "PSS_Social must have density!"
    assert c0.pss_density_adaptive == False, "V0 must NOT have density!"
    assert c1.pss_density_adaptive == True, "V1 must have density!"
    assert cb.beta3_init == 0.0, "Baseline must have no PSS!"
    print("  Density flags: PASS (PSS=T, V0=F, V1=T, BL=none)")

    # ── Test 5: Phi is position-only ──
    import numpy as np
    pss = SocialPSS_Module(c0)
    ego = np.array([0.0, 0.0], dtype=np.float32)
    npc = np.array([[0.25, 0.0]], dtype=np.float32)
    goal = np.array([1.0, 0.0], dtype=np.float32)

    phi = pss.get_phi(ego, npc, goal)
    phi2 = pss.get_phi(ego, npc, goal)
    assert phi == phi2, "Phi not deterministic!"
    print(f"  Phi deterministic: PASS (Phi={phi:.4f})")

    # ── Test 6: Density scaling works ──
    import numpy as np
    npc12 = np.tile(np.array([[0.4, 0.0]]), (12, 1)).astype(np.float32)
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    npc12 = np.stack([0.4*np.cos(angles), 0.4*np.sin(angles)], axis=1).astype(np.float32)

    phi_v0 = SocialPSS_Module(c0).get_phi(ego, npc12, goal)
    phi_v1 = SocialPSS_Module(c1).get_phi(ego, npc12, goal)
    assert phi_v0 != phi_v1, "V0 and V1 Phi should differ (density scaling)!"
    assert abs(phi_v1) < abs(phi_v0), "V1 Phi should be less negative (scaled down)!"
    print(f"  Density scaling effect: PASS (V0 Phi={phi_v0:.4f}, V1 Phi={phi_v1:.4f})")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)