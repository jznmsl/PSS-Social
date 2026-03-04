#!/usr/bin/env python3
"""
run_social.py - Social Navigation Training Runner 

VERSION: 2.2 (Performance Optimized)

Key Features:
  - Variable NPC density: --num-npcs-train and --num-npcs-test
  - Zero-shot generalization testing
  - Triangle test: Baseline vs Safe_Baseline vs PSS_Social
  - Performance optimizations for high-core-count CPUs

Usage:
  # Train with 6 NPCs
  python run_social.py --experiment PSS_Social --num-npcs-train 6 --timesteps 500000

  # Evaluate with 12 NPCs (zero-shot generalization)
  python run_social.py --evaluate ./runs_social/PSS_Social/seed_42 --num-npcs-test 12

  # Full experiment (all 3 configs)
  python run_social.py --full-experiment --seeds 42,123,456

  # FAST MODE for 13900KF + 4090:
  python run_social.py --full-experiment --turbo
"""

from __future__ import annotations

__version__ = "2.3-unified-metrics"

import os
import json
import time
import argparse
import multiprocessing as mp
from datetime import datetime
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# Local imports
from env_social_nav import make_social_nav_env, SocialNavConfig
from pss_social import (
    PSSSocialConfig,
    get_social_experiment_config,
    PSSSocialLocalWrapper,
    PSSSocialGlobalVecWrapper,
)

# ==============================================================================
# Performance Optimization
# ==============================================================================

# Track if PyTorch threads have already been set (can only be done once!)
_PYTORCH_THREADS_SET = False

def optimize_pytorch_for_training(device: str = "cpu", num_envs: int = 16):
    """
    Apply PyTorch optimizations for maximum training speed.

    For CPU (recommended for PPO+MLP):
      - Set optimal thread count (ONLY ONCE!)
      - Enable MKL optimizations
      - Disable gradient debugging

    For GPU:
      - Enable cuDNN autotuning
      - Enable TF32 for faster matmul

    NOTE: Thread settings can only be applied ONCE before any parallel work.
    """
    global _PYTORCH_THREADS_SET

    if device == "cpu":
        # Thread settings can only be set ONCE
        if not _PYTORCH_THREADS_SET:
            try:
                # For CPU-bound training, use fewer threads per process
                physical_cores = mp.cpu_count() // 2  # Assume hyperthreading
                threads_per_env = max(1, physical_cores // num_envs)

                torch.set_num_threads(max(4, threads_per_env * 2))
                torch.set_num_interop_threads(2)
                _PYTORCH_THREADS_SET = True
            except RuntimeError:
                # Already set by another process or call
                pass

        # These can be called multiple times safely
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)

    else:  # GPU
        if torch.cuda.is_available():
            # Enable cuDNN autotuning (finds fastest algorithms)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

            # Enable TF32 for Ampere+ GPUs (RTX 30xx, 40xx)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Disable debug
            torch.autograd.set_detect_anomaly(False)

def get_optimal_settings(cpu_cores: int, gpu_available: bool, timesteps: int) -> dict:
    """
    Return optimal hyperparameters based on hardware.

    For 13900KF (24 cores / 32 threads):
      - Use 24-32 parallel envs
      - Larger batch sizes for efficiency
      - Larger n_steps for fewer updates
    """
    settings = {}

    if cpu_cores >= 24:  # High-end CPU (13900K, 7950X, etc.)
        settings["n_envs"] = 32  # Max out parallelism
        settings["n_steps"] = 2048  # Standard, good balance
        settings["batch_size"] = 256  # Larger batches = faster
        settings["n_epochs"] = 10
        settings["device"] = "cpu"  # CPU is faster for MLP
    elif cpu_cores >= 16:  # Mid-range (12700K, 5800X)
        settings["n_envs"] = 16
        settings["n_steps"] = 2048
        settings["batch_size"] = 128
        settings["n_epochs"] = 10
        settings["device"] = "cpu"
    else:  # Lower-end
        settings["n_envs"] = 8
        settings["n_steps"] = 2048
        settings["batch_size"] = 64
        settings["n_epochs"] = 10
        settings["device"] = "cpu"

    # Ensure batch_size divides n_envs * n_steps evenly
    rollout_size = settings["n_envs"] * settings["n_steps"]
    while rollout_size % settings["batch_size"] != 0:
        settings["batch_size"] //= 2

    return settings

# ==============================================================================
# Hardware Detection
# ==============================================================================

def get_hardware_info() -> dict:
    info = {
        "cpu_cores": mp.cpu_count(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_memory_gb": 0,
    }
    if info["gpu_available"]:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return info

def print_hardware_info(show_turbo: bool = False):
    info = get_hardware_info()
    print("  DETECTED HARDWARE:")
    print(f"   CPU: {info['cpu_cores']} cores")
    if info["gpu_available"]:
        print(f"   GPU: {info['gpu_name']} ({info['gpu_memory_gb']:.1f} GB)")
        print(f"   NOTE: PPO+MLP runs better on CPU. Using --device cpu (default)")
    else:
        print("   GPU: None (using CPU)")

    # Get optimal settings
    optimal = get_optimal_settings(info['cpu_cores'], info['gpu_available'], 10_000_000)

    if show_turbo:
        print(f"\n    TURBO MODE ENABLED:")
        print(f"      n_envs={optimal['n_envs']}, batch_size={optimal['batch_size']}")
        print(f"      n_steps={optimal['n_steps']}, n_epochs={optimal['n_epochs']}")
    else:
        print(f"   Recommended: --n-envs {optimal['n_envs']} --vecenv subproc")
        print(f"   Or use: --turbo (auto-optimize for your hardware)")
    print()

# ==============================================================================
# Time Formatting
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

# ==============================================================================
# Environment Factory
# ==============================================================================

def make_env_fn(
    num_npcs: int,
    scenario: str,
    max_cycles: int,
    seed: int,
    config: PSSSocialConfig,
    use_fir: bool,

    # DENSITY RANDOMIZATION (Ideal 2)

    randomize_density: bool = False,
    min_active_npcs: int = 6,
    max_active_npcs: int = 12,
):
    """Factory function to create social navigation environment."""

    def _init():
        # Create social nav config
        social_config = SocialNavConfig()
        social_config.num_npcs = num_npcs
        social_config.scenario = scenario
        social_config.max_cycles = max_cycles

        # Copy social norm thresholds from FIR config
        social_config.intimate_space = config.intimate_space
        social_config.personal_space = config.personal_space

        # DENSITY RANDOMIZATION (Ideal 2)

        social_config.randomize_density = randomize_density
        social_config.min_active_npcs = min_active_npcs
        social_config.max_active_npcs = max_active_npcs

        # Create environment
        env = make_social_nav_env(
            num_npcs=num_npcs,
            scenario=scenario,
            max_cycles=max_cycles,
            config=social_config,
            randomize_density=randomize_density,
            min_active_npcs=min_active_npcs,
            max_active_npcs=max_active_npcs,
        )

        # Add FIR local wrapper if needed
        if use_fir:
            env = PSSSocialLocalWrapper(env, config)

        # Seed at first reset
        env.reset(seed=seed)
        return env

    return _init

def build_vecenv(vecenv_kind: str, env_fns):
    """Build vectorized environment."""
    if vecenv_kind == "subproc" and len(env_fns) > 1:
        return SubprocVecEnv(env_fns, start_method='spawn')
    return DummyVecEnv(env_fns)

# ==============================================================================
# Training Callback with TensorBoard Logging
# ==============================================================================

class SocialNavCallback(BaseCallback):
    """
    Enhanced callback with:
    - Progress tracking with ETA
    - TensorBoard logging for all metrics
    - Time tracking (time-to-goal, episode length, training speed)
    - Early stopping when success rate is stable (for PSS_Social efficiency)
    """

    def __init__(
        self,
        total_timesteps: int,
        exp_name: str = "",
        seed: int = 0,
        print_interval_pct: int = 2,  # Print every 2% for more frequent updates
        tb_log_interval: int = 1000,  # Log to TB every N steps
        early_stop: bool = True,  # Enable early stopping
        early_stop_min_steps: int = 500_000,  # Minimum steps before early stop
        early_stop_success_threshold: float = 0.98,  # Success rate threshold
        early_stop_patience: int = 5,  # Consecutive checks needed

        # CHECKPOINT & BEST MODEL SAVING 

        run_dir: str = "",                  # Where to save checkpoints
        checkpoint_interval: int = 500_000, # Save checkpoint every N steps
    ):
        super().__init__()
        self.total = int(total_timesteps)
        self.start = None
        self.last_pct = -1
        self.print_interval = print_interval_pct
        self.tb_log_interval = tb_log_interval
        self.fps_history = []
        self.exp_name = exp_name
        self.seed = seed
        self.prefix = f"[{exp_name[:15]:<15} s{seed}]" if exp_name else ""

        # Early Stopping Configuration

        self.early_stop = early_stop
        self.early_stop_min_steps = early_stop_min_steps
        self.early_stop_success_threshold = early_stop_success_threshold
        self.early_stop_patience = early_stop_patience
        self.consecutive_success_checks = 0
        self.early_stopped = False

        # CHECKPOINT & BEST MODEL SAVING

        self.run_dir = run_dir
        self.checkpoint_interval = checkpoint_interval
        self.last_checkpoint_step = 0
        self.best_safe_success = -1.0     # Track best safe_success_rate
        self.best_model_step = 0          # Step at which best model was saved

        # Metric Buffers (for rolling averages)

        self.buffer_size = 100  # Rolling window size

        # Episode-level metrics
        self.ep_collisions = []
        self.ep_intrusions = []
        self.ep_freezing_rates = []
        self.ep_goal_reached = []      # GoalReached: arrived at goal (may have collisions)
        self.ep_safe_success = []      # SafeSuccess: goal_reached AND zero collisions 
        self.ep_lengths = []
        self.ep_rewards = []
        self.ep_time_to_goal = []

        # Density-aware tracking for competence gating (Phase 2.2)
        # Track safe success by NPC density bucket: {bucket: [successes]}
        self.safe_success_by_density = {
            "easy": [],    # 6-8 NPCs
            "medium": [],  # 9-10 NPCs
            "hard": [],    # 11-12+ NPCs
        }

        # Step-level metrics (PSS components)
        self.step_r_pss = []
        self.step_r_int = []
        self.step_r_ext = []  # Extrinsic reward
        self.step_beta3 = []
        self.step_min_dist = []
        self.step_velocity = []

        # Counters
        self.episodes_completed = 0
        self.last_tb_log_step = 0

        # Timing
        self.episode_start_times = {}  # Track individual episode times
        self.total_episode_time = 0.0
        self.ep_wall_times = []  # Wall-clock time per episode

    def _on_training_start(self):
        self.start = time.time()
        self.last_tb_log_step = 0

    def _on_step(self) -> bool:

        # Collect metrics from infos

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])

        for i, info in enumerate(infos):
            if not isinstance(info, dict):
                continue

            # Step-level PSS metrics
            if "r_pss" in info:
                self.step_r_pss.append(info["r_pss"])
            if "r_int" in info:
                self.step_r_int.append(info["r_int"])
            if "r_ext" in info:
                self.step_r_ext.append(info["r_ext"])
            if "beta3" in info:
                self.step_beta3.append(info["beta3"])

            # Social metrics
            if "min_dist_to_npc" in info and np.isfinite(info["min_dist_to_npc"]):
                self.step_min_dist.append(info["min_dist_to_npc"])
            if "ego_velocity" in info:
                self.step_velocity.append(info["ego_velocity"])

            # Episode completion
            if dones[i] if isinstance(dones, (list, np.ndarray)) else dones:
                self.episodes_completed += 1

                # Episode-level metrics
                ep_collisions = info.get("episode_collisions", 0)
                goal_reached = info.get("goal_reached", False)

                if "episode_collisions" in info:
                    self.ep_collisions.append(ep_collisions)
                if "episode_intrusions" in info:
                    self.ep_intrusions.append(info["episode_intrusions"])
                if "freezing_rate" in info:
                    self.ep_freezing_rates.append(info["freezing_rate"])

                # CRITICAL: Track BOTH GoalReached and SafeSuccess
                # SafeSuccess = goal_reached AND episode_collisions == 0
                # This is the PRIMARY METRIC

                self.ep_goal_reached.append(1.0 if goal_reached else 0.0)

                # SafeSuccess: the metric that matters for evaluation!
                safe_success = goal_reached and (ep_collisions == 0)
                self.ep_safe_success.append(1.0 if safe_success else 0.0)

                # Density-aware tracking (for competence gating)
                active_npcs = info.get("active_npc_count", 6)
                if active_npcs <= 8:
                    bucket = "easy"
                elif active_npcs <= 10:
                    bucket = "medium"
                else:
                    bucket = "hard"
                self.safe_success_by_density[bucket].append(1.0 if safe_success else 0.0)
                # Trim density buckets
                for b in self.safe_success_by_density:
                    if len(self.safe_success_by_density[b]) > self.buffer_size:
                        self.safe_success_by_density[b] = self.safe_success_by_density[b][-self.buffer_size:]

                if "episode_steps" in info:
                    self.ep_lengths.append(info["episode_steps"])

                    # Time-to-goal (if goal reached)
                    if goal_reached:
                        self.ep_time_to_goal.append(info["episode_steps"])

                # Episode reward (from SB3's episode info)
                if "episode" in info:
                    self.ep_rewards.append(info["episode"].get("r", 0))

        # Trim buffers to prevent memory growth
        self._trim_buffers()

        # TensorBoard Logging

        if self.num_timesteps - self.last_tb_log_step >= self.tb_log_interval:
            self._log_to_tensorboard()
            self.last_tb_log_step = self.num_timesteps

        # Console Progress Printing

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

            # Recent metrics for console
            recent_col = np.mean(self.ep_collisions[-20:]) if self.ep_collisions else 0
            recent_frz = np.mean(self.ep_freezing_rates[-20:]) if self.ep_freezing_rates else 0
            recent_goal = np.mean(self.ep_goal_reached[-20:]) if self.ep_goal_reached else 0
            recent_safe = np.mean(self.ep_safe_success[-20:]) if self.ep_safe_success else 0
            recent_r_int = np.mean(self.step_r_int[-100:]) if self.step_r_int else 0
            recent_r_ext = np.mean(self.step_r_ext[-100:]) if self.step_r_ext else 0

            # Build the print line with all metrics
            # CRITICAL: Show BOTH Goal% (arrived) and Safe% (arrived + no collision)
            print(f"  {self.prefix} {pct:3d}% | {self.num_timesteps:>9,} | "
                  f"T:{format_time(elapsed)} | FPS:{fps:,.0f} | ETA:{format_time(eta)} | "
                  f"Ep:{self.episodes_completed:,} | "
                  f"Goal:{recent_goal:.0%} | Safe:{recent_safe:.0%} | Col:{recent_col:.2f} | "
                  f"r_ext:{recent_r_ext:+.3f} | Frz:{recent_frz:.0%}",
                  flush=True)  # [FIX] flush for parallel mode

            # Early Stopping Check - NOW USES SafeSuccess (not just goal_reached!)

            if self.early_stop and self.num_timesteps >= self.early_stop_min_steps:
                # Use SAFE SUCCESS for early stopping, not just goal reached!
                # Immediate stop at 100% safe success rate
                if recent_safe >= 1.0:
                    self.early_stopped = True
                    print(f"\n  {self.prefix}  EARLY STOP: SafeSuccess rate 100%!")
                    print(f"  {self.prefix}    Saved {(self.total - self.num_timesteps):,} steps "
                          f"({100*(self.total - self.num_timesteps)/self.total:.0f}% of training)")
                    return False  # Stop training
                # For 98-99%, still wait for patience (optional safety margin)
                elif recent_safe >= self.early_stop_success_threshold:
                    self.consecutive_success_checks += 1
                    if self.consecutive_success_checks >= self.early_stop_patience:
                        self.early_stopped = True
                        print(f"\n  {self.prefix}  EARLY STOP: SafeSuccess rate {recent_safe:.0%} stable for "
                              f"{self.consecutive_success_checks} checks!")
                        print(f"  {self.prefix}    Saved {(self.total - self.num_timesteps):,} steps "
                              f"({100*(self.total - self.num_timesteps)/self.total:.0f}% of training)")
                        return False  # Stop training
                else:
                    self.consecutive_success_checks = 0  # Reset if success drops

        return True

    def _trim_buffers(self):
        """Trim buffers to prevent memory growth."""
        max_size = self.buffer_size * 2

        # Episode buffers
        if len(self.ep_collisions) > max_size:
            self.ep_collisions = self.ep_collisions[-self.buffer_size:]
        if len(self.ep_intrusions) > max_size:
            self.ep_intrusions = self.ep_intrusions[-self.buffer_size:]
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
        if len(self.ep_time_to_goal) > max_size:
            self.ep_time_to_goal = self.ep_time_to_goal[-self.buffer_size:]

        # Step buffers
        if len(self.step_r_pss) > max_size:
            self.step_r_pss = self.step_r_pss[-self.buffer_size:]
        if len(self.step_r_int) > max_size:
            self.step_r_int = self.step_r_int[-self.buffer_size:]
        if len(self.step_r_ext) > max_size:
            self.step_r_ext = self.step_r_ext[-self.buffer_size:]
        if len(self.step_beta3) > max_size:
            self.step_beta3 = self.step_beta3[-self.buffer_size:]
        if len(self.step_min_dist) > max_size:
            self.step_min_dist = self.step_min_dist[-self.buffer_size:]
        if len(self.step_velocity) > max_size:
            self.step_velocity = self.step_velocity[-self.buffer_size:]

    def _log_to_tensorboard(self):
        """Log all metrics to TensorBoard."""
        if self.logger is None:
            return

        # Primary Metrics

        # GoalReached: arrived at goal (may have collisions)
        if self.ep_goal_reached:
            self.logger.record("metrics/goal_reached_rate", np.mean(self.ep_goal_reached[-self.buffer_size:]))

        # SafeSuccess: THE PRIMARY METRIC - goal_reached AND zero collisions!
        if self.ep_safe_success:
            self.logger.record("metrics/safe_success_rate", np.mean(self.ep_safe_success[-self.buffer_size:]))

        # Density-aware SafeSuccess (for analyzing generalization)
        for bucket, values in self.safe_success_by_density.items():
            if values:
                self.logger.record(f"metrics/safe_success_{bucket}", np.mean(values[-50:]))

        if self.ep_collisions:
            # Collision rate: episodes with any collision
            collision_rate = np.mean([1.0 if c > 0 else 0.0 for c in self.ep_collisions[-self.buffer_size:]])
            self.logger.record("metrics/collision_rate", collision_rate)
            self.logger.record("metrics/collisions_per_ep", np.mean(self.ep_collisions[-self.buffer_size:]))

        if self.ep_freezing_rates:
            self.logger.record("metrics/freezing_rate", np.mean(self.ep_freezing_rates[-self.buffer_size:]))

        if self.ep_intrusions:
            self.logger.record("metrics/intrusions_per_ep", np.mean(self.ep_intrusions[-self.buffer_size:]))

        # Time Metrics

        if self.ep_time_to_goal:
            self.logger.record("time/time_to_goal", np.mean(self.ep_time_to_goal[-self.buffer_size:]))
            self.logger.record("time/time_to_goal_min", np.min(self.ep_time_to_goal[-self.buffer_size:]))

        if self.ep_lengths:
            self.logger.record("time/episode_length", np.mean(self.ep_lengths[-self.buffer_size:]))

        # Training time
        elapsed = time.time() - self.start
        self.logger.record("time/wall_clock_minutes", elapsed / 60.0)
        self.logger.record("time/episodes_completed", self.episodes_completed)

        if elapsed > 0:
            self.logger.record("time/episodes_per_minute", self.episodes_completed / (elapsed / 60.0))
            self.logger.record("time/steps_per_second", self.num_timesteps / elapsed)

        # FIR Intrinsic Reward Components

        if self.step_r_pss:
            self.logger.record("fir/r_pss", np.mean(self.step_r_pss[-self.buffer_size:]))
        if self.step_r_int:
            self.logger.record("fir/r_intrinsic_total", np.mean(self.step_r_int[-self.buffer_size:]))
        if self.step_r_ext:
            self.logger.record("fir/r_extrinsic", np.mean(self.step_r_ext[-self.buffer_size:]))

        # Beta scheduling
        if self.step_beta3:
            self.logger.record("fir/beta3_pss", np.mean(self.step_beta3[-10:]))

        # Social Metrics

        if self.step_min_dist:
            self.logger.record("social/min_dist_to_npc", np.mean(self.step_min_dist[-self.buffer_size:]))
        if self.step_velocity:
            self.logger.record("social/avg_velocity", np.mean(self.step_velocity[-self.buffer_size:]))

        # Episode Rewards

        if self.ep_rewards:
            self.logger.record("episode/reward_mean", np.mean(self.ep_rewards[-self.buffer_size:]))
            self.logger.record("episode/reward_std", np.std(self.ep_rewards[-self.buffer_size:]))

        # BEST MODEL + PERIODIC CHECKPOINT SAVING

        if self.run_dir and self.ep_safe_success:
            current_ss = float(np.mean(self.ep_safe_success[-self.buffer_size:]))

            # Save best model (by safe_success_rate)
            if current_ss > self.best_safe_success and self.episodes_completed >= 20:
                self.best_safe_success = current_ss
                self.best_model_step = self.num_timesteps
                try:
                    self.model.save(os.path.join(self.run_dir, "best_model"))
                    # Also save VecNormalize stats for best model
                    if hasattr(self.model, 'get_env') and self.model.get_env() is not None:
                        env = self.model.get_env()
                        if hasattr(env, 'save'):
                            env.save(os.path.join(self.run_dir, "best_vecnorm.pkl"))
                    print(f"  {self.prefix}  New best! SafeSuccess={current_ss:.1%} @ step {self.num_timesteps:,}", flush=True)
                except Exception as e:
                    pass  # Don't crash training on save error

            self.logger.record("checkpoint/best_safe_success", self.best_safe_success)
            self.logger.record("checkpoint/best_model_step", self.best_model_step)

        # Periodic checkpoint (crash insurance)
        if self.run_dir and self.checkpoint_interval > 0:
            if self.num_timesteps - self.last_checkpoint_step >= self.checkpoint_interval:
                self.last_checkpoint_step = self.num_timesteps
                try:
                    ckpt_path = os.path.join(self.run_dir, f"checkpoint_{self.num_timesteps}")
                    self.model.save(ckpt_path)
                    if hasattr(self.model, 'get_env') and self.model.get_env() is not None:
                        env = self.model.get_env()
                        if hasattr(env, 'save'):
                            env.save(os.path.join(self.run_dir, f"checkpoint_{self.num_timesteps}_vecnorm.pkl"))
                    print(f"  {self.prefix}  Checkpoint @ step {self.num_timesteps:,}", flush=True)
                except Exception:
                    pass

    def _on_training_end(self):
        elapsed = time.time() - self.start
        final_fps = self.num_timesteps / max(1e-6, elapsed)

        if self.early_stopped:
            print(f"\n  {self.prefix}  Training Early Stopped!", flush=True)
            steps_saved = self.total - self.num_timesteps
            pct_saved = 100 * steps_saved / self.total
            print(f"     Steps completed: {self.num_timesteps:,} / {self.total:,} ({100*self.num_timesteps/self.total:.0f}%)", flush=True)
            print(f"     Steps saved: {steps_saved:,} ({pct_saved:.0f}%)", flush=True)
        else:
            print(f"\n  {self.prefix}  Training Complete!", flush=True)

        print(f"     Total time: {format_time(elapsed)}", flush=True)
        print(f"     Episodes: {self.episodes_completed}", flush=True)
        print(f"     Avg FPS: {final_fps:,.0f}", flush=True)

        # Show BOTH metrics clearly
        if self.ep_goal_reached:
            print(f"     Final GoalReached Rate: {np.mean(self.ep_goal_reached[-100:]):.1%}", flush=True)
        if self.ep_safe_success:
            print(f"     Final SafeSuccess Rate: {np.mean(self.ep_safe_success[-100:]):.1%} ", flush=True)
        if self.ep_collisions:
            print(f"     Final Col/Ep: {np.mean(self.ep_collisions[-100:]):.2f}", flush=True)
        if self.ep_freezing_rates:
            print(f"     Final Freezing Rate: {np.mean(self.ep_freezing_rates[-100:]):.1%}", flush=True)
        if self.best_safe_success > 0:
            print(f"      Best SafeSuccess: {self.best_safe_success:.1%} @ step {self.best_model_step:,}", flush=True)
            print(f"      best_model.zip saved in {self.run_dir}", flush=True)

# ==============================================================================
# Single Training Run
# ==============================================================================
# Scenario Mixing Helper
# ==============================================================================

def parse_scenario_mix(mix_str: str, n_envs: int) -> List[str]:
    """
    Parse a scenario mix string and allocate envs proportionally.

    Example: "corridor:0.7,intersection:0.3" with n_envs=32
             22 corridor envs + 10 intersection envs

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
        print(f"    scenario-mix fractions sum to {total:.2f}, normalizing to 1.0")
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
# Training
# ==============================================================================

def train_one(
    exp_name: str,
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
    quiet: bool = False,
    early_stop: bool = True,
    early_stop_min_steps: int = 500_000,
    early_stop_patience: int = 5,

    # DENSITY RANDOMIZATION (Ideal 2)

    randomize_density: bool = False,
    min_active_npcs: int = 6,
    max_active_npcs: int = 12,
    turbo: bool = False,  # NEW: Apply PyTorch optimizations
    collision_penalty_override: float = None,  # CLI override for collision penalty

    # SCENARIO MIXING 

    scenario_mix: str = "",  # e.g. "corridor:0.7,intersection:0.3"
) -> Tuple[str, int, str, float]:
    """
    Train one experiment+seed combination.
    Returns: (exp_name, seed, status, elapsed_time)

    NEW: Supports density randomization (Ideal 2) for robust VecNormalize.
    NEW: Supports scenario mixing for multi-scenario training.
    """
    # Apply PyTorch optimizations if turbo mode
    if turbo:
        optimize_pytorch_for_training(device, n_envs)
    run_start = time.time()

    try:
        set_random_seed(seed)

        run_dir = os.path.join(out_dir, exp_name, f"seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)

        # Get FIR config
        config = get_social_experiment_config(exp_name)
        config.max_training_steps = int(timesteps)

        # CLI collision penalty override
        if collision_penalty_override is not None:
            config.collision_penalty = collision_penalty_override
            if not quiet:
                print(f"   [OVERRIDE] collision_penalty = {collision_penalty_override} (from CLI)", flush=True)

        # Decide if FIR modules needed
        # [AUDIT FIX v3] Config-based detection. Old code matched exp_name for
        # "FIR" or "Ablation", which missed PSS_Only_V0-V3 entirely.
        use_fir = (config.beta3_init > 0 or config.beta3_final > 0 or
                   config.collision_penalty > 0)

        if not quiet:
            print(f"\n{''*70}", flush=True)
            print(f" {exp_name} | seed={seed}", flush=True)
            print(f"   num_npcs={num_npcs}, scenario={scenario}, max_cycles={max_cycles}", flush=True)
            print(f"   n_envs={n_envs}, vecenv={vecenv_kind}", flush=True)
            print(f"   timesteps={timesteps:,} | use_fir={use_fir}", flush=True)
            print(f"   beta3_pss={config.beta3_init}, collision_penalty={config.collision_penalty}", flush=True)
            if config.collision_penalty > 0:
                print(f"   collision_penalty={config.collision_penalty}", flush=True)

            # Show density randomization settings

            if randomize_density:
                print(f"    DENSITY RANDOMIZATION: {min_active_npcs}-{max_active_npcs} NPCs per episode", flush=True)
            print(f"     early_stop={early_stop}", flush=True)
            print(f"{''*70}", flush=True)

        # Build environments  with scenario mixing support
        # When scenario_mix is set (e.g. "corridor:0.7,intersection:0.3"),
        # different envs in the VecEnv are assigned different scenarios.
        # This is equivalent to multi-task training with fixed allocation.

        scenario_allocation = parse_scenario_mix(scenario_mix, n_envs) if scenario_mix else []

        if scenario_allocation:
            from collections import Counter
            mix_counts = Counter(scenario_allocation)
            if not quiet:
                mix_str = ", ".join(f"{s}={c}" for s, c in mix_counts.items())
                print(f"    SCENARIO MIX: {mix_str} (total {n_envs} envs)", flush=True)

            env_fns = [
                make_env_fn(
                    num_npcs, scenario_allocation[i], max_cycles, seed + 1000 * i, config, use_fir,
                    randomize_density=randomize_density,
                    min_active_npcs=min_active_npcs,
                    max_active_npcs=max_active_npcs,
                )
                for i in range(n_envs)
            ]
        else:
            env_fns = [
                make_env_fn(
                    num_npcs, scenario, max_cycles, seed + 1000 * i, config, use_fir,
                    randomize_density=randomize_density,
                    min_active_npcs=min_active_npcs,
                    max_active_npcs=max_active_npcs,
                )
                for i in range(n_envs)
            ]
        venv = build_vecenv(vecenv_kind, env_fns)

        # Add global FIR wrapper if needed
        if use_fir:
            pss_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
            venv = PSSSocialGlobalVecWrapper(venv, config, device=pss_device)

        # VecNormalize
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

        # Create PPO model
        tb_log_dir = os.path.join(run_dir, "tb_logs")
        model = PPO(
            "MlpPolicy",
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

        # Save metadata
        meta = {
            "exp_name": exp_name,
            "seed": seed,
            "num_npcs": num_npcs,
            "scenario": scenario,
            "max_cycles": max_cycles,
            "timesteps": timesteps,
            "n_envs": n_envs,
            "vecenv_kind": vecenv_kind,
            "device": device,
            "ppo": {"n_steps": n_steps, "batch_size": batch_size, "n_epochs": n_epochs},
            "fir_config": config.__dict__,
            "timestamp": datetime.now().isoformat(timespec="seconds"),

            # DENSITY RANDOMIZATION (Ideal 2)

            "density_randomization": {
                "enabled": randomize_density,
                "min_active_npcs": min_active_npcs,
                "max_active_npcs": max_active_npcs,
            },
            "scenario_mix": scenario_mix if scenario_mix else "none",
            "early_stop": early_stop,
        }
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Train
        if not quiet:
            print(f"  [{exp_name}]   Training started at {datetime.now().strftime('%H:%M:%S')}...", flush=True)
        callback = SocialNavCallback(
            timesteps, exp_name, seed,
            early_stop=early_stop,
            early_stop_min_steps=early_stop_min_steps,
            early_stop_patience=early_stop_patience,
            run_dir=run_dir,
            checkpoint_interval=500_000,  # Save every 500K steps
        )
        model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)

        # Save model
        model.save(os.path.join(run_dir, "final_model"))
        venv.save(os.path.join(run_dir, "vecnorm.pkl"))
        venv.close()

        elapsed = time.time() - run_start
        return (exp_name, seed, " Success", elapsed)

    except Exception as e:
        elapsed = time.time() - run_start
        import traceback
        traceback.print_exc()
        return (exp_name, seed, f" {str(e)[:50]}", elapsed)

def train_one_wrapper(args):
    """Wrapper for multiprocessing."""
    return train_one(*args)

# ==============================================================================
# Evaluation (Zero-Shot Generalization)
# ==============================================================================

def evaluate_model(
    run_dir: str,
    num_npcs_test: int,
    scenario: str,
    max_cycles: int,
    episodes: int,
    deterministic: bool = True,
) -> dict:
    """
    Evaluate a trained model on a different NPC density (zero-shot generalization).
    """
    from eval_social import evaluate_run

    # Load metadata to get experiment name
    meta_path = os.path.join(run_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        exp_name = meta.get("exp_name", "PSS_Social")
    else:
        exp_name = "PSS_Social"

    return evaluate_run(
        run_dir=run_dir,
        exp_name=exp_name,
        num_npcs=num_npcs_test,
        scenario=scenario,
        max_cycles=max_cycles,
        episodes=episodes,
        deterministic=deterministic,
    )

# ==============================================================================
# Main
# ==============================================================================

def parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]

def main():
    p = argparse.ArgumentParser(
        description="Social Navigation Training ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  python run_social.py --experiment PSS_Social --timesteps 100000 --fast

  # Single experiment with specific config
  python run_social.py --experiment Baseline --num-npcs-train 6 --scenario corridor

  # Full triangle test (Baseline vs Safe_Baseline vs PSS_Social)
  python run_social.py --full-experiment --seeds 42,123,456

  # RECOMMENDED: Density Randomization + Disable Early Stop (Ideal 1 & 2)
  # This is the most robust training configuration for zero-shot generalization

  python run_social.py --full-experiment --randomize-density --no-early-stop \\
      --min-active-npcs 6 --max-active-npcs 16 --timesteps 22000000 \\
      --batch-size 256 --n-epochs 5 --seeds 42,123,456,99

  # With collision penalty override (sweep or custom value)
  python run_social.py --experiment PSS_Social --randomize-density --no-early-stop \\
      --min-active-npcs 6 --max-active-npcs 12 --collision-penalty 5.0 \\
      --timesteps 22000000 --batch-size 256 --n-epochs 5

  # Zero-shot generalization test
  python run_social.py --evaluate ./runs_social/PSS_Social/seed_42 --num-npcs-test 12
        """
    )

    # Mode selection
    p.add_argument("--full-experiment", action="store_true",
                   help="Run full triangle test (Baseline, Safe_Baseline, PSS_Social)")
    p.add_argument("--evaluate", type=str, default="",
                   help="Path to run_dir for evaluation (enables eval mode)")

    # Basic settings
    p.add_argument("--output", type=str, default="./runs_social")
    p.add_argument("--experiment", type=str, action="append", dest="experiments",
                   help="Experiment name(s). Can be specified multiple times: --experiment Baseline --experiment Safe_Baseline")
    p.add_argument("--seeds", type=str, default="42,123,456")

    # Environment settings
    p.add_argument("--num-npcs-train", type=int, default=6,
                   help="Number of NPCs during training")
    p.add_argument("--num-npcs-test", type=int, default=12,
                   help="Number of NPCs for zero-shot test (only used with --evaluate)")
    p.add_argument("--scenario", type=str, default="corridor",
                   choices=["corridor", "intersection", "circle", "random"],
                   help="Scenario type (or base scenario when using --scenario-mix)")
    p.add_argument("--scenario-mix", type=str, default="",
                   help="Mixed scenario training, e.g. 'corridor:0.7,intersection:0.3'. "
                        "Allocates envs proportionally. Overrides --scenario.")
    p.add_argument("--max-cycles", type=int, default=100)

    # Training settings

    # IDEAL 1: Increased training to 10M steps (was 500K)
    # This ensures the model learns the "squeeze" strategies in high-density scenarios

    p.add_argument("--timesteps", type=int, default=10_000_000,
                   help="Total training timesteps (default: 10M for robust training)")
    p.add_argument("--n-envs", type=int, default=16,
                   help="Number of parallel environments (default: 16, use --turbo for auto)")
    p.add_argument("--vecenv", type=str, default="subproc", choices=["dummy", "subproc"],
                   help="VecEnv type (subproc recommended for multi-core)")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device (cpu recommended for PPO+MLP, cuda only helps with CNN)")

    # PPO settings
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)

    # TURBO MODE - Auto-optimize for your hardware

    p.add_argument("--turbo", action="store_true",
                   help=" Auto-optimize settings for maximum speed on your hardware")

    # Fast mode
    p.add_argument("--fast", action="store_true",
                   help="Quick test with fewer steps")

    # Parallel
    p.add_argument("--parallel-seeds", action="store_true",
                   help="Run seeds in parallel (for same experiment)")
    p.add_argument("--parallel-experiments", action="store_true",
                   help="Run all experiments in parallel (Baseline, Safe_Baseline, PSS_Social simultaneously)")
    p.add_argument("--max-parallel", type=int, default=3)

    # Ablation mode
    p.add_argument("--ablation", action="store_true",
                   help="Run ablation experiments (tests each component)")
    p.add_argument("--ablation-full", action="store_true",
                   help="Run ALL ablation experiments (comprehensive)")

    # Early stopping
    p.add_argument("--no-early-stop", action="store_true",
                   help="Disable early stopping (train full timesteps)")
    p.add_argument("--early-stop-min-steps", type=int, default=500_000,
                   help="Minimum steps before early stopping can trigger")
    p.add_argument("--early-stop-patience", type=int, default=5,
                   help="Consecutive success checks needed for early stop")

    # DENSITY RANDOMIZATION (Ideal 2)
    # Randomize NPC count each episode to make VecNormalize robust

    p.add_argument("--randomize-density", action="store_true",
                   help="Enable density randomization (Ideal 2): vary NPC count each episode")

    # [PHASE 2 FIX] Bias training toward hard densities (10-12 NPCs)
    # With uniform 6-12 sampling, N=12 only gets 14% of episodes (1/7)
    # Changed default: 6  10, so now 10-12 NPCs = harder training = better N=12 eval

    # Widen density range to [6,20] so VecNormalize sees the
    # full distribution during training.  Old default [10,12] caused stats
    # to collapse at N16 (distribution shift  observation space mismatch).

    p.add_argument("--min-active-npcs", type=int, default=6,
                   help="Minimum active NPCs when randomizing (default: 6)")
    p.add_argument("--max-active-npcs", type=int, default=16,
                   help="Maximum active NPCs when randomizing (default: 16)")

    # Collision penalty override
    # Allows sweeping cp values from CLI without editing FIR_social.py

    p.add_argument("--collision-penalty", type=float, default=None,
                   help="Override collision_penalty in FIR config (e.g. --collision-penalty 5.0)")

    # Evaluation settings
    p.add_argument("--eval-episodes", type=int, default=50)

    args = p.parse_args()

    # Fast mode adjustments
    if args.fast:
        args.timesteps = min(args.timesteps, 100_000)
        args.n_envs = min(args.n_envs, 2)
        args.seeds = "42"

    # TURBO MODE - Auto-optimize for hardware

    if args.turbo and not args.fast:
        hw = get_hardware_info()
        optimal = get_optimal_settings(hw['cpu_cores'], hw['gpu_available'], args.timesteps)
        args.n_envs = optimal['n_envs']
        args.batch_size = optimal['batch_size']
        args.n_steps = optimal['n_steps']
        args.n_epochs = optimal['n_epochs']
        args.device = optimal['device']
        # Apply PyTorch optimizations
        optimize_pytorch_for_training(args.device, args.n_envs)

    # EVALUATION MODE

    if args.evaluate:
        print("=" * 70)
        print(" ZERO-SHOT GENERALIZATION EVALUATION")
        print("=" * 70)
        print(f"Model: {args.evaluate}")
        print(f"Test NPCs: {args.num_npcs_test} (trained on: check meta.json)")
        print(f"Scenario: {args.scenario}")
        print(f"Episodes: {args.eval_episodes}")
        print("=" * 70)

        results = evaluate_model(
            run_dir=args.evaluate,
            num_npcs_test=args.num_npcs_test,
            scenario=args.scenario,
            max_cycles=args.max_cycles,
            episodes=args.eval_episodes,
        )

        print("\n RESULTS:")
        for key, value in results.items():
            if isinstance(value, float):
                if "rate" in key.lower():
                    print(f"  {key}: {value*100:.1f}%")
                else:
                    print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

        return

    # TRAINING MODE

    t0 = time.time()
    seeds = parse_int_list(args.seeds)

    # Determine experiments to run
    if args.ablation_full:
        # PSS 2x2 factorial ablation + baselines
        experiments = [
            "Baseline", "Safe_Baseline", "PSS_Social",
            "PSS_Only_V0", "PSS_Only_V1", "PSS_Only_V2", "PSS_Only_V3",
        ]
    elif args.ablation:
        # Key ablation: 2x2 factorial (velocity_aware x density_adaptive)
        experiments = [
            "PSS_Social",           # Full method (velocity-aware, no density-adaptive)
            "PSS_Only_V0",          # Neither
            "PSS_Only_V1",          # Velocity-aware only
            "PSS_Only_V2",          # Both
            "PSS_Only_V3",          # Density-adaptive only
        ]
    elif args.full_experiment:
        experiments = ["Baseline", "Safe_Baseline", "PSS_Social"]
    elif args.experiments:
        # User specified one or more --experiment flags
        experiments = args.experiments
    else:
        # Default to PSS_Social if nothing specified
        experiments = ["PSS_Social"]

    # Print header
    print("=" * 80)
    print(" SOCIAL NAVIGATION TRAINING ")
    print(f"   Version: {__version__} - NO BOTTLENECK WALL")
    print("=" * 80)
    print_hardware_info(show_turbo=args.turbo)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {args.output}")
    print()
    print(f" EXPERIMENT PLAN:")
    print(f"   Experiments: {experiments}")
    print(f"   Seeds: {seeds}")
    print(f"   Total runs: {len(experiments) * len(seeds)}")
    print()
    print(f"  SETTINGS:")
    print(f"   num_npcs_train={args.num_npcs_train}, scenario={args.scenario}")
    print(f"   max_cycles={args.max_cycles}, timesteps={args.timesteps:,}")
    print(f"   n_envs={args.n_envs}, vecenv={args.vecenv}, device={args.device}")
    print(f"   batch_size={args.batch_size}, n_steps={args.n_steps}, n_epochs={args.n_epochs}")

    # Show Ideal 1 & 2 settings

    print(f"   early_stop={not args.no_early_stop}")
    if args.randomize_density:
        print(f"    DENSITY RANDOMIZATION (Ideal 2): {args.min_active_npcs}-{args.max_active_npcs} NPCs per episode")
    else:
        print(f"   density_randomization=False (fixed {args.num_npcs_train} NPCs)")
    if args.scenario_mix:
        print(f"    SCENARIO MIX: {args.scenario_mix}")
    if args.collision_penalty is not None:
        print(f"   COLLISION PENALTY OVERRIDE: {args.collision_penalty}")
    if args.turbo:
        print(f"    TURBO MODE: Hardware-optimized settings applied")
    if args.parallel_experiments:
        print(f"    PARALLEL MODE: All experiments run simultaneously per seed")
    elif args.parallel_seeds:
        print(f"    PARALLEL MODE: Seeds run simultaneously per experiment")
    else:
        print(f"    SEQUENTIAL MODE: One run at a time")
    print("=" * 80)

    results = []

    # Build all runs
    all_runs = [(exp, s) for exp in experiments for s in seeds]

    # Handle parallel mode conflicts
    if args.parallel_experiments and args.parallel_seeds:
        print("  Both --parallel-experiments and --parallel-seeds set. Using --parallel-experiments.")
        args.parallel_seeds = False

    if args.parallel_experiments:
        # Parallel experiments mode: Run all experiments for same seed simultaneously
        print(f"\n PARALLEL EXPERIMENTS MODE: Running {len(experiments)} experiments simultaneously per seed")
        print(f"   (Baseline, Safe_Baseline, PSS_Social will train at the same time)")

        # Auto-adjust n_envs to avoid CPU overload
        # Total processes = n_experiments * n_envs, keep under CPU count
        max_total_envs = max(4, mp.cpu_count() - 2)  # Leave 2 cores free
        adjusted_n_envs = max(2, max_total_envs // len(experiments))
        if adjusted_n_envs < args.n_envs:
            print(f"     Reducing n_envs from {args.n_envs} to {adjusted_n_envs} (3 experiments  {adjusted_n_envs} = {3*adjusted_n_envs} total)")
            args.n_envs = adjusted_n_envs

        for seed in seeds:
            print(f"\n{'='*60}")
            print(f" SEED {seed} - Running {len(experiments)} experiments in parallel")
            print(f"{'='*60}")

            run_args = [
                (exp, seed, args.output, args.num_npcs_train, args.scenario,
                 args.max_cycles, args.timesteps, args.n_envs, args.vecenv,
                 args.device, args.n_steps, args.batch_size, args.n_epochs, False,  # quiet=False
                 not args.no_early_stop, args.early_stop_min_steps, args.early_stop_patience,
                 # DENSITY RANDOMIZATION (Ideal 2)
                 args.randomize_density, args.min_active_npcs, args.max_active_npcs,
                 args.turbo,  # turbo flag
                 args.collision_penalty,  # collision penalty override
                 args.scenario_mix)  # scenario mixing
                for exp in experiments
            ]

            with ProcessPoolExecutor(max_workers=len(experiments)) as executor:
                futures = {executor.submit(train_one_wrapper, ra): ra for ra in run_args}

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    exp_name, _, status, elapsed = result
                    status_icon = "[OK]" if "Success" in status else "[FAIL]"
                    print(f"  {status_icon} {exp_name} done in {format_time(elapsed)}")

    elif args.parallel_seeds:
        # Parallel execution
        print(f"\n PARALLEL MODE: Running up to {args.max_parallel} seeds simultaneously")

        for exp in experiments:
            exp_seeds = [s for e, s in all_runs if e == exp]
            print(f"\n   {exp} - {len(exp_seeds)} seeds in parallel")

            run_args = [
                (exp, s, args.output, args.num_npcs_train, args.scenario,
                 args.max_cycles, args.timesteps, args.n_envs, args.vecenv,
                 args.device, args.n_steps, args.batch_size, args.n_epochs, False,  # quiet=False
                 not args.no_early_stop, args.early_stop_min_steps, args.early_stop_patience,
                 # DENSITY RANDOMIZATION (Ideal 2)
                 args.randomize_density, args.min_active_npcs, args.max_active_npcs,
                 args.turbo,  # turbo flag
                 args.collision_penalty,  # collision penalty override
                 args.scenario_mix)  # scenario mixing
                for s in exp_seeds
            ]

            with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
                futures = {executor.submit(train_one_wrapper, ra): ra for ra in run_args}

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    _, seed, status, elapsed = result
                    status_icon = "[OK]" if "Success" in status else "[FAIL]"
                    print(f"    {status_icon} seed={seed} done in {format_time(elapsed)}")
    else:
        # Sequential execution
        for i, (exp, s) in enumerate(all_runs):
            print(f"\n[{i+1}/{len(all_runs)}] {exp} seed={s}")

            result = train_one(
                exp_name=exp,
                seed=s,
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
                quiet=False,
                early_stop=not args.no_early_stop,
                early_stop_min_steps=args.early_stop_min_steps,
                early_stop_patience=args.early_stop_patience,

                # DENSITY RANDOMIZATION (Ideal 2)

                randomize_density=args.randomize_density,
                min_active_npcs=args.min_active_npcs,
                max_active_npcs=args.max_active_npcs,
                turbo=args.turbo,
                collision_penalty_override=args.collision_penalty,
                scenario_mix=args.scenario_mix,
            )
            results.append(result)

    # Summary

    total_time = time.time() - t0

    print("\n" + "=" * 80)
    print(" ALL RUNS COMPLETE!")
    print("=" * 80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {format_time(total_time)}")
    print()

    print(" RESULTS:")
    print(f"  {'Experiment':<20} {'Seed':<8} {'Status':<12} {'Time':<12}")
    print(f"  {'-'*20} {'-'*8} {'-'*12} {'-'*12}")
    for exp, seed, status, elapsed in results:
        status_short = "[OK]" if "Success" in status else "[FAIL]"
        print(f"  {exp:<20} {seed:<8} {status_short:<12} {format_time(elapsed):<12}")

    successful = sum(1 for r in results if "Success" in r[2])
    print(f"\n   Successful: {successful}/{len(results)}")

    print("\n" + "=" * 80)
    print(" NEXT STEPS:")
    print(f"  # Evaluate (same density):")
    print(f"  python eval_social.py {args.output} -n {args.eval_episodes} --num-npcs {args.num_npcs_train}")
    print()
    print(f"  # Zero-shot generalization (higher density):")
    print(f"  python eval_social.py {args.output} -n {args.eval_episodes} --num-npcs {args.num_npcs_test}")
    print()
    print(f"  # TensorBoard:")
    print(f"  tensorboard --logdir {args.output}")
    print("=" * 80)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()