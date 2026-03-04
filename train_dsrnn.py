#!/usr/bin/env python3
"""
Train DS-RNN baseline.

Uses the same environment, reward structure, and evaluation as run_social.py
but replaces MlpPolicy with DS-RNN structural features extractor.

This is a FAIR comparison: same PPO hyperparameters, same observation space,
same reward, same VecNormalize — only the network architecture differs.

Usage:
  # Match your exact Baseline/PSS training settings:
  python train_dsrnn.py --seeds 42,123,456 \
    --randomize-density --min-active-npcs 11 --max-active-npcs 16 \
    --num-npcs-train 16 --timesteps 12000000 \
    --scenario-mix "random:0.5,circle:0.5" \
    --n-envs 32 --vecenv subproc --device cpu \
    --batch-size 256 --n-epochs 5 \
    --output ./runs_dsrnn

  # Quick test
  python train_dsrnn.py --seed 42 --timesteps 50000 --n-envs 2

v2.0
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List
from collections import Counter

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

# Import from your existing codebase
from env_social_nav import make_social_nav_env, SocialNavConfig, FIXED_OBS_DIM, MAX_NPCS
from ds_rnn import get_dsrnn_policy_kwargs, count_parameters, DSRNNFeaturesExtractor


# ==============================================================================
# Helpers (matching run_social.py)
# ==============================================================================

def format_time(seconds: float) -> str:
    if seconds < 0 or not np.isfinite(seconds):
        return "calculating..."
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    elif minutes > 0:
        return f"{minutes}m {secs:02d}s"
    else:
        return f"{secs}s"


def parse_scenario_mix(mix_str: str, n_envs: int) -> List[str]:
    """
    Parse a scenario mix string and allocate envs proportionally.
    Example: "random:0.5,circle:0.5" with n_envs=32 → 16 random + 16 circle
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
        print(f"  ⚠️  scenario-mix fractions sum to {total:.2f}, normalizing to 1.0")
        scenarios = {k: v / total for k, v in scenarios.items()}
    
    allocation = []
    remaining = n_envs
    for i, (name, frac) in enumerate(scenarios.items()):
        if i == len(scenarios) - 1:
            count = remaining
        else:
            count = max(1, round(frac * n_envs))
            remaining -= count
        allocation.extend([name] * count)
    
    return allocation


# ==============================================================================
# Environment Creation
# ==============================================================================

def make_env_fn(
    num_npcs: int,
    scenario: str,
    max_cycles: int,
    seed: int,
    randomize_density: bool = False,
    min_active_npcs: int = 6,
    max_active_npcs: int = 20,
):
    """Create a single env factory function."""
    def _make():
        env = make_social_nav_env(
            num_npcs=num_npcs,
            scenario=scenario,
            max_cycles=max_cycles,
            randomize_density=randomize_density,
            min_active_npcs=min_active_npcs,
            max_active_npcs=max_active_npcs,
        )
        return env
    return _make


def build_vecenv(kind: str, env_fns):
    """Build vectorized environment."""
    if kind == "subproc" and len(env_fns) > 1:
        try:
            return SubprocVecEnv(env_fns, start_method='spawn')
        except Exception as e:
            print(f"  ⚠️ SubprocVecEnv failed ({e}), falling back to DummyVecEnv")
            return DummyVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)


# ==============================================================================
# Training Callback (matching run_social.py SocialNavCallback format)
# ==============================================================================

class DSRNNCallback(BaseCallback):
    """
    Training callback for DS-RNN that matches run_social.py log format exactly.
    
    Collects metrics from training infos (rolling averages) rather than
    standalone evaluations, matching the SocialNavCallback approach.
    """
    
    def __init__(
        self,
        total_timesteps: int,
        exp_name: str = "DS_RNN",
        seed: int = 0,
        run_dir: str = "",
        print_interval_pct: int = 2,
        checkpoint_interval: int = 500_000,
    ):
        super().__init__()
        self.total = int(total_timesteps)
        self.start = None
        self.last_pct = -1
        self.print_interval = print_interval_pct
        self.exp_name = exp_name
        self.seed = seed
        self.prefix = f"[{exp_name[:15]:<15} s{seed}]"
        self.fps_history = []
        
        # Checkpoint & best model saving
        self.run_dir = run_dir
        self.checkpoint_interval = checkpoint_interval
        self.last_checkpoint_step = 0
        self.best_safe_success = -1.0
        self.best_model_step = 0
        
        # Metric buffers (rolling averages, matching run_social.py)
        self.buffer_size = 100
        
        # Episode-level metrics
        self.ep_collisions = []
        self.ep_freezing_rates = []
        self.ep_goal_reached = []
        self.ep_safe_success = []
        self.ep_lengths = []
        self.ep_rewards = []
        
        # Step-level metrics
        self.step_r_ext = []
        
        # Counters
        self.episodes_completed = 0
    
    def _on_training_start(self):
        self.start = time.time()
    
    def _on_step(self) -> bool:
        # ── Collect metrics from infos ──
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        
        for i, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            
            # Step-level: extrinsic reward
            if "r_ext" in info:
                self.step_r_ext.append(info["r_ext"])
            
            # Episode completion
            if dones[i] if isinstance(dones, (list, np.ndarray)) else dones:
                self.episodes_completed += 1
                
                ep_collisions = info.get("episode_collisions", 0)
                goal_reached = info.get("goal_reached", False)
                
                if "episode_collisions" in info:
                    self.ep_collisions.append(ep_collisions)
                if "freezing_rate" in info:
                    self.ep_freezing_rates.append(info["freezing_rate"])
                
                self.ep_goal_reached.append(1.0 if goal_reached else 0.0)
                
                # SafeSuccess: goal_reached AND zero collisions 
                safe_success = goal_reached and (ep_collisions == 0)
                self.ep_safe_success.append(1.0 if safe_success else 0.0)
                
                if "episode_steps" in info:
                    self.ep_lengths.append(info["episode_steps"])
                
                if "episode" in info:
                    self.ep_rewards.append(info["episode"].get("r", 0))
        
        # Trim buffers
        self._trim_buffers()
        
        # ── Console Progress (every print_interval_pct %) ──
        pct = int(self.num_timesteps / max(1, self.total) * 100)
        
        if pct >= self.last_pct + self.print_interval:
            self.last_pct = pct
            elapsed = time.time() - self.start
            fps = self.num_timesteps / max(1e-6, elapsed)
            
            self.fps_history.append(fps)
            if len(self.fps_history) > 10:
                self.fps_history.pop(0)
            smoothed_fps = np.mean(self.fps_history)
            
            remaining = self.total - self.num_timesteps
            eta = remaining / max(1, smoothed_fps)
            
            # Recent metrics (rolling window of 20 episodes)
            recent_col = np.mean(self.ep_collisions[-20:]) if self.ep_collisions else 0
            recent_frz = np.mean(self.ep_freezing_rates[-20:]) if self.ep_freezing_rates else 0
            recent_goal = np.mean(self.ep_goal_reached[-20:]) if self.ep_goal_reached else 0
            recent_safe = np.mean(self.ep_safe_success[-20:]) if self.ep_safe_success else 0
            recent_r_ext = np.mean(self.step_r_ext[-100:]) if self.step_r_ext else 0
            
            print(f"  {self.prefix} {pct:3d}% | {self.num_timesteps:>9,} | "
                  f"T:{format_time(elapsed)} | FPS:{fps:,.0f} | ETA:{format_time(eta)} | "
                  f"Ep:{self.episodes_completed:,} | "
                  f"Goal:{recent_goal:.0%} | Safe:{recent_safe:.0%} | Col:{recent_col:.2f} | "
                  f"r_ext:{recent_r_ext:+.3f} | Frz:{recent_frz:.0%}",
                  flush=True)
        
        # ── Best Model Saving (based on rolling SafeSuccess) ──
        if self.ep_safe_success and self.episodes_completed >= 20:
            current_ss = np.mean(self.ep_safe_success[-100:])
            if current_ss > self.best_safe_success:
                self.best_safe_success = current_ss
                self.best_model_step = self.num_timesteps
                if self.run_dir:
                    try:
                        self.model.save(os.path.join(self.run_dir, "best_model"))
                        env = self.model.get_env()
                        if hasattr(env, 'save'):
                            env.save(os.path.join(self.run_dir, "best_vecnorm.pkl"))
                    except Exception:
                        pass
                    print(f"  {self.prefix} 🏆 New best! SafeSuccess={current_ss:.1%} @ step {self.num_timesteps:,}", flush=True)
        
        # ── Periodic Checkpoint ──
        if self.num_timesteps - self.last_checkpoint_step >= self.checkpoint_interval:
            self.last_checkpoint_step = self.num_timesteps
            if self.run_dir:
                try:
                    self.model.save(os.path.join(self.run_dir, f"checkpoint_{self.num_timesteps}"))
                    env = self.model.get_env()
                    if hasattr(env, 'save'):
                        env.save(os.path.join(self.run_dir, f"checkpoint_{self.num_timesteps}_vecnorm.pkl"))
                    print(f"  {self.prefix} 💾 Checkpoint @ step {self.num_timesteps:,}", flush=True)
                except Exception:
                    pass
        
        return True
    
    def _trim_buffers(self):
        max_size = self.buffer_size * 10
        if len(self.ep_collisions) > max_size:
            self.ep_collisions = self.ep_collisions[-self.buffer_size:]
        if len(self.ep_freezing_rates) > max_size:
            self.ep_freezing_rates = self.ep_freezing_rates[-self.buffer_size:]
        if len(self.ep_goal_reached) > max_size:
            self.ep_goal_reached = self.ep_goal_reached[-self.buffer_size:]
        if len(self.ep_safe_success) > max_size:
            self.ep_safe_success = self.ep_safe_success[-self.buffer_size:]
        if len(self.ep_lengths) > max_size:
            self.ep_lengths = self.ep_lengths[-self.buffer_size:]
        if len(self.ep_rewards) > max_size:
            self.ep_rewards = self.ep_rewards[-self.buffer_size:]
        if len(self.step_r_ext) > max_size:
            self.step_r_ext = self.step_r_ext[-self.buffer_size:]
    
    def _on_training_end(self):
        elapsed = time.time() - self.start
        final_fps = self.num_timesteps / max(1e-6, elapsed)
        
        print(f"\n  {self.prefix} ✅ Training Complete!", flush=True)
        print(f"     Total time: {format_time(elapsed)}", flush=True)
        print(f"     Episodes: {self.episodes_completed}", flush=True)
        print(f"     Avg FPS: {final_fps:,.0f}", flush=True)
        
        if self.ep_goal_reached:
            print(f"     Final GoalReached Rate: {np.mean(self.ep_goal_reached[-100:]):.1%}", flush=True)
        if self.ep_safe_success:
            print(f"     Final SafeSuccess Rate: {np.mean(self.ep_safe_success[-100:]):.1%} ", flush=True)
        if self.ep_collisions:
            print(f"     Final Col/Ep: {np.mean(self.ep_collisions[-100:]):.2f}", flush=True)
        if self.ep_freezing_rates:
            print(f"     Final Freezing Rate: {np.mean(self.ep_freezing_rates[-100:]):.1%}", flush=True)
        if self.best_safe_success > 0:
            print(f"     🏆 Best SafeSuccess: {self.best_safe_success:.1%} @ step {self.best_model_step:,}", flush=True)
            print(f"     📂 best_model.zip saved in {self.run_dir}", flush=True)


# ==============================================================================
# Training
# ==============================================================================

def train_dsrnn(
    seed: int = 42,
    num_npcs: int = 16,
    scenario: str = "random",
    scenario_mix: str = "",
    max_cycles: int = 100,
    timesteps: int = 12_000_000,
    n_envs: int = 32,
    vecenv_kind: str = "subproc",
    device: str = "cpu",
    out_dir: str = "./runs_dsrnn",
    exp_name: str = "DS_RNN",
    # PPO hyperparameters
    n_steps: int = 2048,
    batch_size: int = 256,
    n_epochs: int = 5,
    learning_rate: float = 3e-4,
    # DS-RNN architecture
    neighbor_embed_dim: int = 64,
    ego_embed_dim: int = 64,
    social_embed_dim: int = 64,
    n_attention_heads: int = 4,
    features_dim: int = 128,
    # Density randomization
    randomize_density: bool = False,
    min_active_npcs: int = 6,
    max_active_npcs: int = 16,
):
    """Train DS-RNN agent with same setup as Baseline PPO."""
    
    run_dir = os.path.join(out_dir, exp_name, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    
    set_random_seed(seed)
    
    # ── Run header (matching run_social.py format) ──
    sep = "━" * 70
    print(f"\n{sep}")
    print(f"🚀 {exp_name} | seed={seed}")
    print(f"   num_npcs={num_npcs}, scenario={scenario}, max_cycles={max_cycles}")
    print(f"   n_envs={n_envs}, vecenv={vecenv_kind}")
    print(f"   timesteps={timesteps:,} | method=DS-RNN (structural attention)")
    print(f"   PPO: n_steps={n_steps}, batch={batch_size}, epochs={n_epochs}, lr={learning_rate}")
    if randomize_density:
        print(f"   🎲 DENSITY RANDOMIZATION: {min_active_npcs}-{max_active_npcs} NPCs per episode")
    if scenario_mix:
        print(f"   ⏱️  early_stop=False")
    print(sep)
    
    # ── Create environments ──
    scenario_allocation = parse_scenario_mix(scenario_mix, n_envs) if scenario_mix else []
    
    if scenario_allocation:
        mix_counts = Counter(scenario_allocation)
        mix_str = ", ".join(f"{s}={c}" for s, c in mix_counts.items())
        print(f"   🎰 SCENARIO MIX: {mix_str} (total {n_envs} envs)", flush=True)
        
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
        env_fns = [
            make_env_fn(
                num_npcs, scenario, max_cycles, seed + 1000 * i,
                randomize_density=randomize_density,
                min_active_npcs=min_active_npcs,
                max_active_npcs=max_active_npcs,
            )
            for i in range(n_envs)
        ]
    
    venv = build_vecenv(vecenv_kind, env_fns)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # ── Create PPO model with DS-RNN features extractor ──
    policy_kwargs = get_dsrnn_policy_kwargs(
        neighbor_embed_dim=neighbor_embed_dim,
        ego_embed_dim=ego_embed_dim,
        social_embed_dim=social_embed_dim,
        n_attention_heads=n_attention_heads,
        features_dim=features_dim,
    )
    
    tb_log_dir = os.path.join(run_dir, "tb_logs")
    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
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
        policy_kwargs=policy_kwargs,
    )
    
    # Report architecture
    params = count_parameters(model)
    print(f"   Architecture: {params['total']:,} params "
          f"(extractor={params['features_extractor']:,}, heads={params['policy_heads']:,})")
    
    # ── Save metadata ──
    meta = {
        "exp_name": exp_name,
        "method": "DS-RNN",
        "seed": seed,
        "num_npcs": num_npcs,
        "scenario": scenario,
        "scenario_mix": scenario_mix if scenario_mix else "none",
        "max_cycles": max_cycles,
        "timesteps": timesteps,
        "n_envs": n_envs,
        "vecenv_kind": vecenv_kind,
        "device": device,
        "ppo": {
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "learning_rate": learning_rate,
        },
        "dsrnn": {
            "neighbor_embed_dim": neighbor_embed_dim,
            "ego_embed_dim": ego_embed_dim,
            "social_embed_dim": social_embed_dim,
            "n_attention_heads": n_attention_heads,
            "features_dim": features_dim,
        },
        "density_randomization": {
            "enabled": randomize_density,
            "min_active_npcs": min_active_npcs,
            "max_active_npcs": max_active_npcs,
        },
        "parameters": params,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    # ── Training callback (matching run_social.py format) ──
    callback = DSRNNCallback(
        total_timesteps=timesteps,
        exp_name=exp_name,
        seed=seed,
        run_dir=run_dir,
        print_interval_pct=2,
        checkpoint_interval=500_000,
    )
    
    # ── Train ──
    print(f"  [{exp_name}] ⏱️  Training started at {datetime.now().strftime('%H:%M:%S')}...", flush=True)
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
    
    # ── Save final model ──
    model.save(os.path.join(run_dir, "final_model"))
    venv.save(os.path.join(run_dir, "vecnorm.pkl"))
    venv.close()
    
    return run_dir, callback


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train DS-RNN baseline ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Match your PSS/Baseline training exactly:
  python train_dsrnn.py --seeds 42,123,456 \\
    --randomize-density --min-active-npcs 11 --max-active-npcs 16 \\
    --num-npcs-train 16 --timesteps 12000000 \\
    --scenario-mix "random:0.5,circle:0.5" \\
    --n-envs 32 --vecenv subproc --device cpu \\
    --batch-size 256 --n-epochs 5 \\
    --output ./runs_dsrnn

  # Quick test
  python train_dsrnn.py --seed 42 --timesteps 100000 --n-envs 2
        """
    )
    
    # ── Seeds ──
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seed", type=int, help="Single random seed")
    group.add_argument("--seeds", type=str, help="Comma-separated seeds (e.g., 42,123,456)")
    
    # ── Environment (matching run_social.py arg names) ──
    parser.add_argument("--num-npcs-train", type=int, default=16)
    parser.add_argument("--scenario", type=str, default="random",
                        choices=["corridor", "intersection", "circle", "random"])
    parser.add_argument("--scenario-mix", type=str, default="",
                        help="Scenario mixing, e.g. 'random:0.5,circle:0.5'")
    parser.add_argument("--max-cycles", type=int, default=100)
    
    # ── Training ──
    parser.add_argument("--timesteps", type=int, default=12_000_000)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--vecenv", type=str, default="subproc", choices=["dummy", "subproc"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="./runs_dsrnn")
    parser.add_argument("--exp-name", type=str, default="DS_RNN")
    
    # ── PPO hyperparameters ──
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    
    # ── DS-RNN architecture ──
    parser.add_argument("--neighbor-embed", type=int, default=64)
    parser.add_argument("--ego-embed", type=int, default=64)
    parser.add_argument("--social-embed", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--features-dim", type=int, default=128)
    
    # ── Density randomization ──
    parser.add_argument("--randomize-density", action="store_true")
    parser.add_argument("--min-active-npcs", type=int, default=6)
    parser.add_argument("--max-active-npcs", type=int, default=16)
    
    # ── Compatibility flags ──
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Ignored. For CLI compatibility with run_social.py.")
    
    args = parser.parse_args()
    
    # Determine seeds
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [42]
    
    # ── Pipeline header (matching run_social.py) ──
    print("=" * 80)
    print(f"🧠 DS-RNN SOCIAL NAVIGATION TRAINING ")
    print(f"   Version: 2.0 - Structural Attention Baseline")
    print("=" * 80)
    print(f"\nStart: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {args.output}")
    print(f"\n📋 EXPERIMENT PLAN:")
    print(f"   Experiments: ['{args.exp_name}']")
    print(f"   Seeds: {seeds}")
    print(f"   Total runs: {len(seeds)}")
    print(f"\n⚙️  SETTINGS:")
    print(f"   num_npcs_train={args.num_npcs_train}, scenario={args.scenario}")
    print(f"   max_cycles={args.max_cycles}, timesteps={args.timesteps:,}")
    print(f"   n_envs={args.n_envs}, vecenv={args.vecenv}, device={args.device}")
    print(f"   batch_size={args.batch_size}, n_steps={args.n_steps}, n_epochs={args.n_epochs}")
    print(f"   early_stop=False")
    if args.randomize_density:
        print(f"   🎲 DENSITY RANDOMIZATION: {args.min_active_npcs}-{args.max_active_npcs} NPCs per episode")
    if args.scenario_mix:
        print(f"   🎰 SCENARIO MIX: {args.scenario_mix}")
    print(f"   ⏳ SEQUENTIAL MODE: One run at a time")
    print("=" * 80)
    
    results = []
    for idx, seed in enumerate(seeds):
        print(f"\n[{idx+1}/{len(seeds)}] {args.exp_name} seed={seed}\n")
        try:
            run_dir, callback = train_dsrnn(
                seed=seed,
                num_npcs=args.num_npcs_train,
                scenario=args.scenario,
                scenario_mix=args.scenario_mix,
                max_cycles=args.max_cycles,
                timesteps=args.timesteps,
                n_envs=args.n_envs,
                vecenv_kind=args.vecenv,
                device=args.device,
                out_dir=args.output,
                exp_name=args.exp_name,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                learning_rate=args.lr,
                neighbor_embed_dim=args.neighbor_embed,
                ego_embed_dim=args.ego_embed,
                social_embed_dim=args.social_embed,
                n_attention_heads=args.attention_heads,
                features_dim=args.features_dim,
                randomize_density=args.randomize_density,
                min_active_npcs=args.min_active_npcs,
                max_active_npcs=args.max_active_npcs,
            )
            results.append((seed, "✅ Success", run_dir,
                            callback.best_safe_success, callback.best_model_step))
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append((seed, f"❌ {str(e)[:60]}", "", 0, 0))
    
    # ── Summary ──
    print(f"\n{'=' * 80}")
    print(f"  DS-RNN Training Summary")
    print(f"{'=' * 80}")
    for seed, status, path, best_ss, best_step in results:
        print(f"  Seed {seed}: {status}")
        if path:
            print(f"    Best SafeSuccess: {best_ss:.1%} @ step {best_step:,}")
            print(f"    → {path}")
    
    print(f"\nNext steps:")
    print(f"  python eval_dsrnn.py {args.output}/{args.exp_name} --densities 11,13,15,17,19,21,23 --scenario random")
    print(f"  python eval_dsrnn.py {args.output}/{args.exp_name} --densities 11,13,15,17,19,21,23 --scenario circle")


if __name__ == "__main__":
    main()