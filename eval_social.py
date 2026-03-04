#!/usr/bin/env python3
"""
eval_social.py - Social Navigation Evaluation Script 

VERSION: 2.2 (Proper evaluation with SAFE SUCCESS metric)

Key Features:
  - Load trained PPO models from runs_social directory
  - Evaluate with different NPC counts (zero-shot generalization)
  - Report metrics: SUCCESS rate, SAFE SUCCESS rate, collision rate, freezing rate

IMPORTANT METRICS:
  - SafeSuccess%: Goal reached AND zero collisions (primary metric)
  - Success%: Goal reached (may include episodes with collisions)

Usage:
  # Evaluate specific model
  python eval_social.py ./runs_social/PSS_Social/seed_42 -n 50 --num-npcs 6
  
  # Zero-shot test (trained on 6 NPCs, test on 12)
  python eval_social.py ./runs_social/PSS_Social/seed_42 -n 100 --num-npcs 12
  
  # Compare all experiments
  python eval_social.py ./runs_social --compare -n 50
"""

from __future__ import annotations

__version__ = "2.3-jerk-metric"

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Local imports - IMPORTANT: uses the FIXED env_social_nav.py
from env_social_nav import make_social_nav_env, SocialNavConfig, FIXED_OBS_DIM, MAX_NPCS


@dataclass
class EvalResults:
    """Results from evaluation run."""
    experiment: str = ""
    seed: int = 0
    num_episodes: int = 0
    num_npcs_train: int = 6
    num_npcs_eval: int = 6
    
    # Metrics
    success_rate: float = 0.0          # Goal reached (may have collisions)
    safe_success_rate: float = 0.0     # Goal reached AND zero collisions
    collision_rate: float = 0.0
    collision_per_episode: float = 0.0
    intrusion_rate: float = 0.0
    freezing_rate: float = 0.0
    avg_episode_length: float = 0.0
    avg_reward: float = 0.0
    
    # ══════════════════════════════════════════════════════════════════════════
    # [PHASE 3] Jerk metric - measures smoothness of motion
    # High jerk = jerky stop-and-go behavior (Safe_Baseline weakness)
    # Low jerk = smooth socially comfortable motion (PSS_Social strength)
    # ══════════════════════════════════════════════════════════════════════════
    avg_jerk: float = 0.0              # Average action smoothness (lower = better)
    
    # Additional stats
    successes: int = 0
    safe_successes: int = 0            # NEW: Goal reached + zero collisions
    collisions_total: int = 0
    intrusions_total: int = 0
    freezes_total: int = 0
    jerk_total: float = 0.0            # Total jerk accumulated


def load_model_and_env(
    model_path: str,
    num_npcs: int = 6,
    scenario: str = "corridor",
    max_cycles: int = 100,
) -> Tuple[PPO, VecNormalize]:
    """
    Load trained PPO model and create evaluation environment.
    
    Args:
        model_path: Path to model directory (containing best_model.zip and vecnormalize.pkl)
        num_npcs: Number of NPCs for evaluation
        scenario: "corridor" or "intersection"
        max_cycles: Maximum steps per episode
        
    Returns:
        Tuple of (model, vec_env)
    """
    model_path = Path(model_path)
    
    # Find model file
    model_file = model_path / "best_model.zip"
    if not model_file.exists():
        model_file = model_path / "final_model.zip"
    if not model_file.exists():
        # Try to find any .zip file
        zip_files = list(model_path.glob("*.zip"))
        if zip_files:
            model_file = zip_files[0]
        else:
            raise FileNotFoundError(f"No model file found in {model_path}")
    
    # Find VecNormalize stats (check both common filenames)
    # If using best_model, prefer best_vecnorm.pkl (matched stats)
    vecnorm_file = None
    if "best_model" in str(model_file):
        candidate = model_path / "best_vecnorm.pkl"
        if candidate.exists():
            vecnorm_file = candidate
    if vecnorm_file is None:
        vecnorm_file = model_path / "vecnorm.pkl"
    if not vecnorm_file.exists():
        vecnorm_file = model_path / "vecnormalize.pkl"
    
    # Create environment
    def make_env():
        return make_social_nav_env(
            num_npcs=num_npcs,
            scenario=scenario,
            max_cycles=max_cycles,
        )
    
    vec_env = DummyVecEnv([make_env])
    
    # Load VecNormalize if exists
    if vecnorm_file.exists():
        vec_env = VecNormalize.load(str(vecnorm_file), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print(f"  ⚠️ No VecNormalize found at {vecnorm_file}")
    
    # Load model
    model = PPO.load(str(model_file), env=vec_env)
    
    return model, vec_env


def evaluate_model(
    model: PPO,
    vec_env: VecNormalize,
    num_episodes: int = 50,
    deterministic: bool = True,
    verbose: bool = False,
) -> EvalResults:
    """
    Evaluate a trained model.
    
    Args:
        model: Trained PPO model
        vec_env: Vectorized environment with normalization
        num_episodes: Number of episodes to evaluate
        deterministic: Use deterministic actions
        verbose: Print per-episode info
        
    Returns:
        EvalResults with metrics
    """
    results = EvalResults(num_episodes=num_episodes)
    
    episode_rewards = []
    episode_lengths = []
    
    episodes_done = 0
    obs = vec_env.reset()
    
    episode_reward = 0
    episode_length = 0
    episode_collisions = 0
    episode_intrusions = 0
    episode_freezes = 0
    
    # ══════════════════════════════════════════════════════════════════════════
    # [PHASE 3] Jerk tracking - measures smoothness of motion
    # Jerk = mean(|action_t - action_{t-1}|) over episode
    # Lower jerk = smoother motion = more socially acceptable
    # ══════════════════════════════════════════════════════════════════════════
    prev_action = None
    total_jerk = 0.0
    total_action_steps = 0
    
    while episodes_done < num_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = vec_env.step(action)
        
        episode_reward += reward[0]
        episode_length += 1
        
        # Track jerk (action smoothness)
        if prev_action is not None:
            jerk = float(np.mean(np.abs(action[0] - prev_action)))
            total_jerk += jerk
            total_action_steps += 1
        prev_action = action[0].copy()
        
        # Track per-step metrics
        info_dict = info[0]
        if info_dict.get("collisions", 0) > 0:
            episode_collisions += info_dict["collisions"]
        if info_dict.get("intrusions", 0) > 0:
            episode_intrusions += info_dict["intrusions"]
        if info_dict.get("freezing", False):
            episode_freezes += 1
        
        if done[0]:
            # Episode finished
            episodes_done += 1
            
            # Check success (goal reached, may have collisions)
            goal_reached = info_dict.get("goal_reached", False)
            if goal_reached:
                results.successes += 1
            
            # Check SAFE success (goal reached AND zero collisions)
            # This is the metric that matters !
            if goal_reached and episode_collisions == 0:
                results.safe_successes += 1
            
            # Accumulate totals
            results.collisions_total += episode_collisions
            results.intrusions_total += episode_intrusions
            results.freezes_total += episode_freezes
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if verbose and episodes_done % 10 == 0:
                safe_marker = "🏆" if (goal_reached and episode_collisions == 0) else ("✓" if goal_reached else "✗")
                print(f"  Episode {episodes_done}/{num_episodes}: "
                      f"reward={episode_reward:.2f}, len={episode_length}, "
                      f"col={episode_collisions}, "
                      f"success={safe_marker}")
            
            # Reset counters (but keep prev_action for jerk continuity)
            episode_reward = 0
            episode_length = 0
            episode_collisions = 0
            episode_intrusions = 0
            episode_freezes = 0
            prev_action = None  # Reset at episode boundary
    
    # Calculate final metrics
    results.success_rate = results.successes / num_episodes
    results.safe_success_rate = results.safe_successes / num_episodes  # NEW!
    results.collision_per_episode = results.collisions_total / num_episodes
    results.collision_rate = 1.0 if results.collisions_total > 0 else 0.0
    results.collision_rate = min(1.0, results.collisions_total / (num_episodes * 10))  # Normalized
    results.intrusion_rate = min(1.0, results.intrusions_total / (num_episodes * 10))
    results.freezing_rate = results.freezes_total / sum(episode_lengths) if episode_lengths else 0.0
    results.avg_episode_length = np.mean(episode_lengths) if episode_lengths else 0.0
    results.avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
    
    # [PHASE 3] Calculate average jerk
    results.jerk_total = total_jerk
    results.avg_jerk = total_jerk / max(1, total_action_steps)
    
    return results


# ==============================================================================
# evaluate_run: High-level entry point (used by run_social.py --evaluate)
# ==============================================================================

def evaluate_run(
    run_dir: str,
    exp_name: str = "PSS_Social",
    num_npcs: int = 6,
    scenario: str = "corridor",
    max_cycles: int = 100,
    episodes: int = 50,
    deterministic: bool = True,
) -> dict:
    """
    Load a trained model from run_dir and evaluate it.
    
    This is the bridge function called by run_social.py --evaluate.
    It loads the model+env, runs evaluation, and returns a plain dict.
    
    Args:
        run_dir: Path to model directory (containing best_model.zip/final_model.zip + vecnorm.pkl)
        exp_name: Experiment name (for logging)
        num_npcs: Number of NPCs for evaluation
        scenario: "corridor" or "intersection"
        max_cycles: Maximum steps per episode
        episodes: Number of evaluation episodes
        deterministic: Use deterministic actions
        
    Returns:
        Dict of metric_name -> value
    """
    print(f"\n  📦 Loading model from {run_dir}")
    print(f"     exp={exp_name}, num_npcs={num_npcs}, scenario={scenario}")
    
    model, vec_env = load_model_and_env(
        model_path=run_dir,
        num_npcs=num_npcs,
        scenario=scenario,
        max_cycles=max_cycles,
    )
    
    print(f"  🔬 Evaluating {episodes} episodes (deterministic={deterministic})...")
    
    results = evaluate_model(
        model=model,
        vec_env=vec_env,
        num_episodes=episodes,
        deterministic=deterministic,
    )
    
    # Set metadata
    results.experiment = exp_name
    results.num_npcs_eval = num_npcs
    
    # Close env
    vec_env.close()
    
    # Convert EvalResults dataclass to dict for run_social.py
    from dataclasses import asdict
    return asdict(results)


def find_experiments(runs_dir: Path) -> Dict[str, List[Path]]:
    """
    Find all experiments in the runs directory.
    
    Returns:
        Dict mapping experiment name to list of seed directories
    """
    experiments = defaultdict(list)
    
    for exp_dir in runs_dir.iterdir():
        if exp_dir.is_dir():
            exp_name = exp_dir.name
            
            # Find seed directories
            for seed_dir in exp_dir.iterdir():
                if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
                    # Check if it has a model
                    if (seed_dir / "best_model.zip").exists() or \
                       (seed_dir / "final_model.zip").exists() or \
                       list(seed_dir.glob("*.zip")):
                        experiments[exp_name].append(seed_dir)
    
    return dict(experiments)


def print_results_table(all_results: Dict[str, List[EvalResults]], num_npcs: int):
    """Print a nice comparison table."""
    print("\n" + "=" * 90)
    print(f"EVALUATION RESULTS (NPCs={num_npcs})")
    print("=" * 100)
    
    # Header - SafeSuccess% is the KEY metric !
    # Jerk shows motion smoothness (lower = smoother = better social navigation)
    print(f"{'Experiment':<18} {'Seed':<6} {'SafeSuccess%':<12} {'Success%':<10} {'Col/Ep':<8} "
          f"{'Freeze%':<9} {'AvgLen':<8} {'Jerk':<8}")
    print("-" * 100)
    
    for exp_name, results_list in sorted(all_results.items()):
        for r in results_list:
            # Highlight safe success - this is the metric that matters!
            safe_str = f"{r.safe_success_rate*100:>8.1f}%"
            print(f"{exp_name:<18} {r.seed:<6} {safe_str:<12} {r.success_rate*100:>7.1f}%  "
                  f"{r.collision_per_episode:>6.2f}  {r.freezing_rate*100:>7.1f}%  "
                  f"{r.avg_episode_length:>6.1f}  {r.avg_jerk:>6.3f}")
        
        # Print mean across seeds
        if len(results_list) > 1:
            mean_safe = np.mean([r.safe_success_rate for r in results_list]) * 100
            mean_success = np.mean([r.success_rate for r in results_list]) * 100
            mean_col = np.mean([r.collision_per_episode for r in results_list])
            mean_freeze = np.mean([r.freezing_rate for r in results_list]) * 100
            mean_len = np.mean([r.avg_episode_length for r in results_list])
            mean_jerk = np.mean([r.avg_jerk for r in results_list])
            
            std_safe = np.std([r.safe_success_rate for r in results_list]) * 100
            std_success = np.std([r.success_rate for r in results_list]) * 100
            
            print(f"{'  └─ MEAN':<18} {'---':<6} {mean_safe:>5.1f}±{std_safe:.1f}%   "
                  f"{mean_success:>5.1f}±{std_success:.1f}%  "
                  f"{mean_col:>6.2f}  {mean_freeze:>7.1f}%  "
                  f"{mean_len:>6.1f}  {mean_jerk:>6.3f}")
        print()
    
    print("=" * 100)
    print("NOTE: SafeSuccess% = Goal reached AND zero collisions (primary metric)")
    print("      Success% = Goal reached (may include episodes with collisions)")
    print("      Jerk = Action smoothness (lower = smoother motion = more socially comfortable)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Social Navigation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate specific model
  python eval_social.py ./runs_social/PSS_Social/seed_42 -n 50
  
  # Zero-shot test (trained on 6 NPCs, test on 12)
  python eval_social.py ./runs_social/PSS_Social/seed_42 -n 100 --num-npcs 12
  
  # Compare all experiments
  python eval_social.py ./runs_social --compare -n 50
        """
    )
    
    parser.add_argument("path", type=str,
                        help="Path to model directory or runs_social for comparison")
    parser.add_argument("-n", "--num-episodes", type=int, default=50,
                        help="Number of episodes to evaluate (default: 50)")
    parser.add_argument("--num-npcs", type=int, default=6,
                        help="Number of NPCs for evaluation (default: 6)")
    parser.add_argument("--scenario", type=str, default="corridor",
                        choices=["corridor", "intersection", "circle", "random"],
                        help="Evaluation scenario (default: corridor)")
    parser.add_argument("--max-cycles", type=int, default=100,
                        help="Maximum steps per episode (default: 100)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all experiments in the directory")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-episode details")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic actions (default: True)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic actions")
    
    args = parser.parse_args()
    
    if args.stochastic:
        args.deterministic = False
    
    path = Path(args.path)
    
    print(f"\n{'='*60}")
    print(f"Social Navigation Evaluation")
    print(f"{'='*60}")
    print(f"📐 Fixed observation dimension: {FIXED_OBS_DIM}")
    print(f"   (Max NPCs: {MAX_NPCS})")
    print(f"🎯 Evaluating with {args.num_npcs} NPCs")
    print(f"📊 Episodes per model: {args.num_episodes}")
    print()
    
    all_results = {}
    
    if args.compare:
        # Find all experiments
        experiments = find_experiments(path)
        
        if not experiments:
            print(f"❌ No experiments found in {path}")
            print("   Expected structure: {path}/{{experiment_name}}/seed_{{N}}/best_model.zip")
            sys.exit(1)
        
        print(f"Found {len(experiments)} experiments:")
        for exp_name, seed_dirs in experiments.items():
            print(f"  - {exp_name}: {len(seed_dirs)} seeds")
        print()
        
        for exp_name, seed_dirs in sorted(experiments.items()):
            all_results[exp_name] = []
            
            for seed_dir in sorted(seed_dirs):
                seed = int(seed_dir.name.split("_")[1])
                
                print(f"Evaluating {exp_name} (seed {seed})...")
                
                try:
                    model, vec_env = load_model_and_env(
                        seed_dir,
                        num_npcs=args.num_npcs,
                        scenario=args.scenario,
                        max_cycles=args.max_cycles,
                    )
                    
                    results = evaluate_model(
                        model, vec_env,
                        num_episodes=args.num_episodes,
                        deterministic=args.deterministic,
                        verbose=args.verbose,
                    )
                    
                    results.experiment = exp_name
                    results.seed = seed
                    results.num_npcs_eval = args.num_npcs
                    
                    all_results[exp_name].append(results)
                    
                    vec_env.close()
                    
                except Exception as e:
                    print(f"  ❌ Error: {e}")
        
        # Print comparison table
        print_results_table(all_results, args.num_npcs)
        
    else:
        # Evaluate single model
        if not path.exists():
            print(f"❌ Path not found: {path}")
            sys.exit(1)
        
        # Determine experiment name and seed from path
        exp_name = path.parent.name if path.parent.name != "runs_social" else path.name
        seed = 0
        if path.name.startswith("seed_"):
            seed = int(path.name.split("_")[1])
        
        print(f"Evaluating: {path}")
        print()
        
        try:
            model, vec_env = load_model_and_env(
                path,
                num_npcs=args.num_npcs,
                scenario=args.scenario,
                max_cycles=args.max_cycles,
            )
            
            results = evaluate_model(
                model, vec_env,
                num_episodes=args.num_episodes,
                deterministic=args.deterministic,
                verbose=args.verbose,
            )
            
            results.experiment = exp_name
            results.seed = seed
            results.num_npcs_eval = args.num_npcs
            
            all_results[exp_name] = [results]
            
            vec_env.close()
            
            print_results_table(all_results, args.num_npcs)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    print("✅ Evaluation complete!")


# ==============================================================================
# Quick Environment Test (for debugging)
# ==============================================================================

def test_environment():
    """Quick test to verify environment works."""
    print("\n" + "=" * 60)
    print("🧪 Environment Quick Test")
    print("=" * 60)
    print(f"📐 FIXED observation dimension: {FIXED_OBS_DIM}")
    
    # Test different NPC counts
    test_configs = [4, 6, 8, 12, 16]
    obs_shapes = []
    
    for num_npcs in test_configs:
        env = make_social_nav_env(num_npcs=num_npcs, scenario="corridor", max_cycles=50)
        obs, _ = env.reset(seed=42)
        obs_shapes.append(obs.shape)
        print(f"  {num_npcs:2d} NPCs → obs shape: {obs.shape}")
        env.close()
    
    all_same = all(s == obs_shapes[0] for s in obs_shapes)
    
    print()
    if all_same:
        print(f"✅ Zero-shot compatibility VERIFIED!")
        print(f"   All observation shapes: {obs_shapes[0]}")
    else:
        print(f"❌ Zero-shot compatibility BROKEN!")
        print(f"   Shapes: {obs_shapes}")
    
    return all_same


if __name__ == "__main__":
    # If no arguments, run quick test
    if len(sys.argv) == 1:
        test_environment()
        print("\n💡 Usage: python eval_social.py ./runs_social --compare -n 50")
    else:
        main()