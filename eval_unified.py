"""
Unified Evaluation.

Evaluates ALL agents (PPO variants, SFM, ORCA, SARL, DS-RNN)
across density sweeps and scenarios with identical metrics.

Outputs CSV for plotting.

Usage:
  # Full density sweep (random scenario)
  python eval_unified.py ./runs_social --densities 11,13,15,17,19,21,23 --scenario random

  # Quick test
  python eval_unified.py ./runs_social --densities 11,17,23 -n 20

  # Skip specific baselines
  python eval_unified.py ./runs_social --no-orca --no-dsrnn --densities 11,17,23

  # Custom SARL seeds
  python eval_unified.py ./runs_social --sarl-seeds 9,10 --densities 11,17,23

v1.1 - Unified evaluation (all baselines)
"""

__version__ = "1.1"

import argparse
import sys
import os
import csv
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, fields, asdict
from collections import defaultdict

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


from env_social_nav import make_social_nav_env, SocialNavConfig, FIXED_OBS_DIM, MAX_NPCS

try:
    from policies_analytic import SFMPolicy, ORCAPolicy, RVO2_AVAILABLE
except ImportError:
    RVO2_AVAILABLE = False
    print("⚠️ policies_analytic.py not found. Only RL agents will be evaluated.")

# SARL baseline (requires train_baselines.py in path)
try:
    from train_baselines import SARLFeatureExtractor
    SARL_AVAILABLE = True
except ImportError:
    SARL_AVAILABLE = False

# DS-RNN baseline (optional)
try:
    from ds_rnn import DSRNNFeaturesExtractor
    DSRNN_AVAILABLE = True
except ImportError:
    DSRNN_AVAILABLE = False

# LSTM-RL baseline (requires sb3-contrib)
try:
    from sb3_contrib import RecurrentPPO
    LSTM_RL_AVAILABLE = True
except ImportError:
    LSTM_RL_AVAILABLE = False


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class EvalRow:
    """One row of evaluation results (one agent × one density × one seed/run)."""
    agent: str = ""
    density: int = 0
    scenario: str = "corridor"
    seed: int = 0
    
    success_rate: float = 0.0
    safe_success_rate: float = 0.0
    collision_per_episode: float = 0.0
    avg_episode_length: float = 0.0
    avg_jerk: float = 0.0
    avg_min_dist: float = 0.0
    freezing_rate: float = 0.0
    
    num_episodes: int = 0


# ==============================================================================
# Core Evaluation Loop (works for ANY agent type)
# ==============================================================================

def evaluate_agent(
    agent,
    vec_env,
    num_episodes: int = 50,
    deterministic: bool = True,
    verbose: bool = False,
) -> EvalRow:
    """
    Evaluate any agent (PPO model or AnalyticPolicy) using VecEnv interface.
    
    Both PPO and analytic agents implement .predict(obs) -> (action, state).
    """
    row = EvalRow(num_episodes=num_episodes)
    
    episode_rewards = []
    episode_lengths = []
    episode_collisions_list = []
    episode_min_dists = []
    
    episodes_done = 0
    obs = vec_env.reset()
    
    ep_reward = 0.0
    ep_length = 0
    ep_collisions = 0
    ep_intrusions = 0
    ep_freezes = 0
    ep_min_dist = float('inf')
    
    # Jerk tracking
    prev_action = None
    total_jerk = 0.0
    total_jerk_steps = 0
    
    safe_successes = 0
    successes = 0
    total_freezes = 0
    total_steps = 0
    
    while episodes_done < num_episodes:
        action, _ = agent.predict(obs, deterministic=deterministic)
        obs, reward, done, info = vec_env.step(action)
        
        ep_reward += reward[0]
        ep_length += 1
        total_steps += 1
        
        # Jerk (action smoothness)
        if prev_action is not None:
            jerk = float(np.mean(np.abs(action[0] - prev_action)))
            total_jerk += jerk
            total_jerk_steps += 1
        prev_action = action[0].copy()
        
        # Per-step metrics
        info_dict = info[0] if isinstance(info, (list, tuple)) else info
        
        if info_dict.get("collisions", 0) > 0:
            ep_collisions += info_dict["collisions"]
        if info_dict.get("intrusions", 0) > 0:
            ep_intrusions += info_dict["intrusions"]
        if info_dict.get("freezing", False):
            ep_freezes += 1
            total_freezes += 1
        
        min_d = info_dict.get("min_dist_to_npc", float('inf'))
        if np.isfinite(min_d):
            ep_min_dist = min(ep_min_dist, min_d)
        
        if done[0]:
            episodes_done += 1
            
            goal_reached = info_dict.get("goal_reached", False)
            if goal_reached:
                successes += 1
            if goal_reached and ep_collisions == 0:
                safe_successes += 1
            
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            episode_collisions_list.append(ep_collisions)
            episode_min_dists.append(ep_min_dist if np.isfinite(ep_min_dist) else 0.0)
            
            if verbose and episodes_done % 10 == 0:
                marker = "🏆" if (goal_reached and ep_collisions == 0) else ("✓" if goal_reached else "✗")
                print(f"    Ep {episodes_done}/{num_episodes}: "
                      f"col={ep_collisions} min_d={ep_min_dist:.2f} {marker}")
            
            # Reset episode counters
            ep_reward = 0.0
            ep_length = 0
            ep_collisions = 0
            ep_intrusions = 0
            ep_freezes = 0
            ep_min_dist = float('inf')
            prev_action = None
    
    # Compute final metrics
    # [AUDIT FIX] Use episodes_done (actual) not num_episodes (requested)
    row.num_episodes = episodes_done
    row.success_rate = successes / max(1, episodes_done)
    row.safe_success_rate = safe_successes / max(1, episodes_done)
    row.collision_per_episode = np.mean(episode_collisions_list) if episode_collisions_list else 0.0
    row.avg_episode_length = np.mean(episode_lengths) if episode_lengths else 0.0
    row.avg_jerk = total_jerk / max(1, total_jerk_steps)
    row.avg_min_dist = np.mean(episode_min_dists) if episode_min_dists else 0.0
    row.freezing_rate = total_freezes / max(1, total_steps)
    
    return row


# ==============================================================================
# Recurrent Agent Evaluation (LSTM_RL)
# ==============================================================================

def evaluate_recurrent_agent(
    model,
    vec_env,
    num_episodes: int = 50,
    deterministic: bool = True,
) -> EvalRow:
    """
    Evaluate a RecurrentPPO (LSTM) agent.

    RecurrentPPO requires explicit LSTM state management across steps.

    NOTE: DummyVecEnv returns the terminal episode's info when done=True
    (it does NOT wrap it in "terminal_info").
    NOTE: Scalar per-episode accumulators assume num_envs=1.
    """
    assert vec_env.num_envs == 1, (
        f"evaluate_recurrent_agent requires num_envs=1, got {vec_env.num_envs}. "
        f"Per-episode accumulators would mix across environments."
    )
    episodes_done = 0
    obs = vec_env.reset()

    # Initialize LSTM states
    lstm_states = None
    episode_starts = np.ones((vec_env.num_envs,), dtype=bool)

    # Per-episode accumulators
    ep_collisions = 0
    ep_min_dist = float('inf')
    ep_length = 0
    prev_action = None

    # Global accumulators
    successes = 0
    safe_successes = 0
    all_collisions = []
    all_lengths = []
    all_min_dists = []
    total_jerk = 0.0
    total_jerk_steps = 0
    total_freezes = 0
    total_steps = 0

    while episodes_done < num_episodes:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        obs, rewards, dones, infos = vec_env.step(action)
        episode_starts = dones

        for i in range(vec_env.num_envs):
            total_steps += 1
            ep_length += 1
            info = infos[i] if isinstance(infos, (list, tuple, np.ndarray)) else infos

            if prev_action is not None:
                jerk = float(np.mean(np.abs(action[i] - prev_action)))
                total_jerk += jerk
                total_jerk_steps += 1
            prev_action = action[i].copy()

            if info.get("collisions", 0) > 0:
                ep_collisions += info["collisions"]
            if info.get("freezing", False):
                total_freezes += 1

            min_d = info.get("min_dist_to_npc", float('inf'))
            if np.isfinite(min_d):
                ep_min_dist = min(ep_min_dist, min_d)

            if dones[i]:
                episodes_done += 1
                goal = info.get("goal_reached", False)

                if goal:
                    successes += 1
                if goal and ep_collisions == 0:
                    safe_successes += 1

                all_collisions.append(ep_collisions)
                all_lengths.append(ep_length)
                all_min_dists.append(
                    ep_min_dist if np.isfinite(ep_min_dist) else 0.0
                )

                ep_collisions = 0
                ep_min_dist = float('inf')
                ep_length = 0
                prev_action = None

                if episodes_done >= num_episodes:
                    break

    row = EvalRow()
    row.num_episodes = episodes_done
    row.success_rate = successes / max(1, episodes_done)
    row.safe_success_rate = safe_successes / max(1, episodes_done)
    row.collision_per_episode = float(np.mean(all_collisions)) if all_collisions else 0.0
    row.avg_episode_length = float(np.mean(all_lengths)) if all_lengths else 0.0
    row.avg_jerk = total_jerk / max(1, total_jerk_steps)
    row.avg_min_dist = float(np.mean(all_min_dists)) if all_min_dists else 0.0
    row.freezing_rate = total_freezes / max(1, total_steps)
    return row


# ==============================================================================
# Environment Creation Helpers
# ==============================================================================

def make_eval_env(num_npcs: int, scenario: str, max_cycles: int = 100):
    """Create a raw evaluation environment (no VecNormalize)."""
    def _make():
        return make_social_nav_env(
            num_npcs=num_npcs,
            scenario=scenario,
            max_cycles=max_cycles,
        )
    return DummyVecEnv([_make])


def make_eval_env_with_vecnorm(
    model_path: Path,
    num_npcs: int,
    scenario: str,
    max_cycles: int = 100,
) -> Tuple[PPO, VecNormalize]:
    """Load PPO model with VecNormalize for evaluation."""
    # Find model file
    model_file = model_path / "best_model.zip"
    if not model_file.exists():
        model_file = model_path / "final_model.zip"
    if not model_file.exists():
        zip_files = list(model_path.glob("*.zip"))
        if zip_files:
            model_file = zip_files[0]
        else:
            raise FileNotFoundError(f"No model in {model_path}")
    
    # Find VecNormalize (prefer best_vecnorm.pkl when loading best_model)
    vecnorm_file = None
    if "best_model" in str(model_file):
        candidate = model_path / "best_vecnorm.pkl"
        if candidate.exists():
            vecnorm_file = candidate
    if vecnorm_file is None:
        vecnorm_file = model_path / "vecnorm.pkl"
    if not vecnorm_file.exists():
        vecnorm_file = model_path / "vecnormalize.pkl"
    
    # Create env
    def _make():
        return make_social_nav_env(
            num_npcs=num_npcs,
            scenario=scenario,
            max_cycles=max_cycles,
        )
    
    vec_env = DummyVecEnv([_make])
    
    if vecnorm_file.exists():
        vec_env = VecNormalize.load(str(vecnorm_file), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    # Log which checkpoint was loaded
    vn_name = vecnorm_file.name if vecnorm_file.exists() else "NONE"
    print(f"    Loading: {model_file.name} + {vn_name}", flush=True)

    model = PPO.load(str(model_file), env=vec_env)
    return model, vec_env


def load_baseline_model(
    seed_dir: Path,
    num_npcs: int,
    scenario: str,
    max_cycles: int = 100,
    agent_type: str = "SARL",
):
    """
    Load a baseline model (SARL, DS-RNN, or LSTM_RL) that needs
    custom_objects or a different model class (RecurrentPPO).

    Returns (model, vec_env).
    """
    import json

    # Read training max_cycles from meta.json if available
    meta_path = seed_dir / "meta.json"
    train_max_cycles = max_cycles
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        train_max_cycles = meta.get("max_cycles", max_cycles)

    # Find model file
    model_file = seed_dir / "best_model.zip"
    if not model_file.exists():
        model_file = seed_dir / "final_model.zip"
    if not model_file.exists():
        zip_files = list(seed_dir.glob("*.zip"))
        if zip_files:
            model_file = zip_files[0]
        else:
            raise FileNotFoundError(f"No model in {seed_dir}")

    # Find VecNormalize
    vecnorm_file = None
    if "best_model" in str(model_file):
        candidate = seed_dir / "best_vecnorm.pkl"
        if candidate.exists():
            vecnorm_file = candidate
    if vecnorm_file is None:
        vecnorm_file = seed_dir / "vecnorm.pkl"
    if not vecnorm_file.exists():
        vecnorm_file = seed_dir / "vecnormalize.pkl"

    # Create env
    def _make():
        return make_social_nav_env(
            num_npcs=num_npcs,
            scenario=scenario,
            max_cycles=train_max_cycles,
        )

    vec_env = DummyVecEnv([_make])

    if vecnorm_file.exists():
        vec_env = VecNormalize.load(str(vecnorm_file), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    vn_name = vecnorm_file.name if vecnorm_file.exists() else "NONE"
    print(f"    Loading: {model_file.name} + {vn_name} ({agent_type})", flush=True)

    # Load model based on agent type
    if agent_type == "LSTM_RL":
        if not LSTM_RL_AVAILABLE:
            raise ImportError("sb3-contrib required to load LSTM_RL models")
        model = RecurrentPPO.load(str(model_file), env=vec_env)
    else:
        custom_objects = {}
        if agent_type == "DS_RNN" and DSRNN_AVAILABLE:
            custom_objects["DSRNNFeaturesExtractor"] = DSRNNFeaturesExtractor
        elif agent_type == "SARL" and SARL_AVAILABLE:
            # SB3 pickles policy_kwargs with Python code objects, which break
            # across Python versions (e.g. 3.11 -> 3.12 "code() argument 13
            # must be str, not int"). Passing policy_kwargs in custom_objects
            # bypasses deserialization entirely.
            custom_objects["policy_kwargs"] = dict(
                features_extractor_class=SARLFeatureExtractor,
                features_extractor_kwargs=dict(
                    embed_dim=64,
                    num_heads=4,
                    features_dim=64,
                ),
                net_arch=dict(pi=[64], vf=[64]),
            )
        model = PPO.load(str(model_file), env=vec_env, custom_objects=custom_objects)

    return model, vec_env


def find_baseline_seeds(
    runs_dir: Path,
    agent_name: str,
    seed_filter: List[int] = None,
) -> List[Path]:
    """
    Find seed directories for a specific baseline agent.

    Args:
        seed_filter: If given, only return seed dirs whose seed number
                     is in this list.  E.g. [9, 10] for SARL.
    """
    agent_dir = runs_dir / agent_name
    if not agent_dir.exists():
        return []

    seed_dirs = []
    for d in sorted(agent_dir.iterdir()):
        if not (d.is_dir() and d.name.startswith("seed_")):
            continue
        if not ((d / "best_model.zip").exists() or
                (d / "final_model.zip").exists()):
            continue

        if seed_filter is not None:
            seed_num = int(d.name.split("_")[1])
            if seed_num not in seed_filter:
                continue

        seed_dirs.append(d)

    return seed_dirs


# ==============================================================================
# Experiment Discovery
# ==============================================================================

def find_rl_experiments(runs_dir: Path) -> Dict[str, List[Path]]:
    """
    Find all RL experiments (Baseline, Safe_Baseline, PSS_Social, etc.).

    Excludes SARL and DS_RNN which are handled by dedicated baseline
    loading code (load_baseline_model) that passes the correct
    custom_objects for their feature extractors.
    """
    # Agents handled by load_baseline_model (need custom_objects or
    # special deserialization). Adding them here avoids duplicate
    # evaluation and PPO.load failures for custom architectures.
    BASELINE_AGENTS = {"SARL", "DS_RNN", "LSTM_RL"}

    experiments = defaultdict(list)
    
    if not runs_dir.exists():
        return dict(experiments)
    
    for exp_dir in runs_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name not in BASELINE_AGENTS:
            for seed_dir in exp_dir.iterdir():
                if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
                    if (seed_dir / "best_model.zip").exists() or \
                       (seed_dir / "final_model.zip").exists() or \
                       list(seed_dir.glob("*.zip")):
                        experiments[exp_dir.name].append(seed_dir)
    
    return dict(experiments)


# ==============================================================================
# Main Sweep
# ==============================================================================

def run_density_sweep(
    runs_dir: Path,
    densities: List[int],
    scenario: str = "corridor",
    num_episodes: int = 50,
    max_cycles: int = 100,
    include_sfm: bool = True,
    include_orca: bool = True,
    include_sarl: bool = True,
    include_dsrnn: bool = True,
    include_lstm_rl: bool = True,
    analytic_seeds: List[int] = None,
    sarl_seeds: List[int] = None,
    dsrnn_seeds: List[int] = None,
    lstm_rl_seeds: List[int] = None,
    verbose: bool = False,
) -> List[EvalRow]:
    """
    Run full density sweep for all agents.
    
    Args:
        analytic_seeds: Seeds for SFM/ORCA env randomness (default: [42, 123, 456, 9, 10]).
        sarl_seeds:    Which SARL seed dirs to evaluate (default: None = all).
        dsrnn_seeds:   Which DS-RNN seed dirs to evaluate (default: None = all).
        lstm_rl_seeds: Which LSTM_RL seed dirs to evaluate (default: None = all).
    
    Returns list of EvalRow for CSV export.
    """
    if analytic_seeds is None:
        analytic_seeds = [42, 123, 456, 9, 10]
    
    all_rows = []
    
    # Discover RL experiments
    rl_experiments = find_rl_experiments(runs_dir)
    
    if not rl_experiments:
        print(f"⚠️ No RL experiments found in {runs_dir}")
        print(f"   Expected: {runs_dir}/{{Baseline,PSS_Social,...}}/seed_*/best_model.zip")
    else:
        print(f"Found RL experiments: {list(rl_experiments.keys())}")
    
    # Check analytic agents
    if include_orca and not RVO2_AVAILABLE:
        print("⚠️ ORCA disabled (Python-RVO2 not installed)")
        include_orca = False
    if include_sarl and not SARL_AVAILABLE:
        print("⚠️ SARL disabled (train_baselines.py not importable)")
        include_sarl = False
    if include_dsrnn and not DSRNN_AVAILABLE:
        print("⚠️ DS-RNN disabled (ds_rnn.py not importable)")
        include_dsrnn = False
    if include_lstm_rl and not LSTM_RL_AVAILABLE:
        print("⚠️ LSTM_RL disabled (sb3-contrib not installed)")
        include_lstm_rl = False
    
    print(f"\nDensity sweep: {densities}")
    print(f"Scenario: {scenario}")
    print(f"Episodes per eval: {num_episodes}")
    print()
    
    for n_npcs in densities:
        print(f"\n{'='*70}")
        print(f"  DENSITY N = {n_npcs}")
        print(f"{'='*70}")
        
        # ── RL Agents ──
        for exp_name, seed_dirs in sorted(rl_experiments.items()):
            for seed_dir in sorted(seed_dirs):
                seed = int(seed_dir.name.split("_")[1])
                print(f"  📊 {exp_name} (seed {seed}) @ N={n_npcs}...", end=" ", flush=True)
                
                try:
                    model, vec_env = make_eval_env_with_vecnorm(
                        seed_dir, n_npcs, scenario, max_cycles
                    )
                    
                    row = evaluate_agent(model, vec_env, num_episodes, verbose=verbose)
                    row.agent = exp_name
                    row.density = n_npcs
                    row.scenario = scenario
                    row.seed = seed
                    
                    all_rows.append(row)
                    print(f"SafeSuccess={row.safe_success_rate*100:.0f}% "
                          f"Col={row.collision_per_episode:.2f} "
                          f"Jerk={row.avg_jerk:.3f}")
                    
                    vec_env.close()
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        # ── SFM Agent (run with multiple env seeds for fair variance) ──
        if include_sfm:
            for env_seed in analytic_seeds:
                print(f"  📊 SFM (seed {env_seed}) @ N={n_npcs}...", end=" ", flush=True)
                try:
                    np.random.seed(env_seed)
                    vec_env = make_eval_env(n_npcs, scenario, max_cycles)
                    sfm = SFMPolicy(vec_env)
                    
                    row = evaluate_agent(sfm, vec_env, num_episodes, verbose=verbose)
                    row.agent = "SFM"
                    row.density = n_npcs
                    row.scenario = scenario
                    row.seed = env_seed
                    
                    all_rows.append(row)
                    print(f"SafeSuccess={row.safe_success_rate*100:.0f}% "
                          f"Col={row.collision_per_episode:.2f} "
                          f"Jerk={row.avg_jerk:.3f}")
                    
                    vec_env.close()
                except Exception as e:
                    print(f"❌ SFM Error: {e}")
        
        # ── ORCA Agent (run with multiple env seeds for fair variance) ──
        if include_orca:
            for env_seed in analytic_seeds:
                print(f"  📊 ORCA (seed {env_seed}) @ N={n_npcs}...", end=" ", flush=True)
                try:
                    np.random.seed(env_seed)
                    vec_env = make_eval_env(n_npcs, scenario, max_cycles)
                    orca = ORCAPolicy(vec_env)
                    
                    row = evaluate_agent(orca, vec_env, num_episodes, verbose=verbose)
                    row.agent = "ORCA"
                    row.density = n_npcs
                    row.scenario = scenario
                    row.seed = env_seed
                    
                    all_rows.append(row)
                    print(f"SafeSuccess={row.safe_success_rate*100:.0f}% "
                          f"Col={row.collision_per_episode:.2f} "
                          f"Jerk={row.avg_jerk:.3f}")
                    
                    vec_env.close()
                except Exception as e:
                    print(f"❌ ORCA Error: {e}")
        
        # ── SARL Agent (RL baseline with attention feature extractor) ──
        if include_sarl and SARL_AVAILABLE:
            sarl_seed_dirs = find_baseline_seeds(runs_dir, "SARL", sarl_seeds)
            for seed_dir in sarl_seed_dirs:
                seed = int(seed_dir.name.split("_")[1])
                print(f"  📊 SARL (seed {seed}) @ N={n_npcs}...", end=" ", flush=True)
                try:
                    model, vec_env = load_baseline_model(
                        seed_dir, n_npcs, scenario, max_cycles,
                        agent_type="SARL",
                    )
                    
                    row = evaluate_agent(model, vec_env, num_episodes, verbose=verbose)
                    row.agent = "SARL"
                    row.density = n_npcs
                    row.scenario = scenario
                    row.seed = seed
                    
                    all_rows.append(row)
                    print(f"SafeSuccess={row.safe_success_rate*100:.0f}% "
                          f"Col={row.collision_per_episode:.2f} "
                          f"Jerk={row.avg_jerk:.3f}")
                    
                    vec_env.close()
                except Exception as e:
                    print(f"❌ SARL Error: {e}")
        
        # ── DS-RNN Agent ──
        if include_dsrnn and DSRNN_AVAILABLE:
            dsrnn_seed_dirs = find_baseline_seeds(runs_dir, "DS_RNN", dsrnn_seeds)
            for seed_dir in dsrnn_seed_dirs:
                seed = int(seed_dir.name.split("_")[1])
                print(f"  📊 DS_RNN (seed {seed}) @ N={n_npcs}...", end=" ", flush=True)
                try:
                    model, vec_env = load_baseline_model(
                        seed_dir, n_npcs, scenario, max_cycles,
                        agent_type="DS_RNN",
                    )
                    
                    row = evaluate_agent(model, vec_env, num_episodes, verbose=verbose)
                    row.agent = "DS_RNN"
                    row.density = n_npcs
                    row.scenario = scenario
                    row.seed = seed
                    
                    all_rows.append(row)
                    print(f"SafeSuccess={row.safe_success_rate*100:.0f}% "
                          f"Col={row.collision_per_episode:.2f} "
                          f"Jerk={row.avg_jerk:.3f}")
                    
                    vec_env.close()
                except Exception as e:
                    print(f"❌ DS_RNN Error: {e}")
        
        # ── LSTM_RL Agent (recurrent policy, needs special eval loop) ──
        if include_lstm_rl and LSTM_RL_AVAILABLE:
            lstm_seed_dirs = find_baseline_seeds(runs_dir, "LSTM_RL", lstm_rl_seeds)
            for seed_dir in lstm_seed_dirs:
                seed = int(seed_dir.name.split("_")[1])
                print(f"  📊 LSTM_RL (seed {seed}) @ N={n_npcs}...", end=" ", flush=True)
                try:
                    model, vec_env = load_baseline_model(
                        seed_dir, n_npcs, scenario, max_cycles,
                        agent_type="LSTM_RL",
                    )
                    
                    row = evaluate_recurrent_agent(model, vec_env, num_episodes)
                    row.agent = "LSTM_RL"
                    row.density = n_npcs
                    row.scenario = scenario
                    row.seed = seed
                    
                    all_rows.append(row)
                    print(f"SafeSuccess={row.safe_success_rate*100:.0f}% "
                          f"Col={row.collision_per_episode:.2f} "
                          f"Jerk={row.avg_jerk:.3f}")
                    
                    vec_env.close()
                except Exception as e:
                    print(f"❌ LSTM_RL Error: {e}")
    
    return all_rows


# ==============================================================================
# CSV Export
# ==============================================================================

def save_results_csv(rows: List[EvalRow], output_path: str, append: bool = False):
    """Save results to CSV for plotting. If append=True, add to existing file."""
    if not rows:
        print("No results to save!")
        return
    
    field_names = [f.name for f in fields(EvalRow)]
    
    if append and os.path.exists(output_path):
        # Append without re-writing header
        with open(output_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            for row in rows:
                writer.writerow(asdict(row))
        print(f"\n💾 Appended {len(rows)} rows to: {output_path}")
    else:
        # Write fresh file with header
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
        print(f"\n💾 Results saved to: {output_path}")


# ==============================================================================
# Results Table
# ==============================================================================

def print_results_table(rows: List[EvalRow]):
    """Print a comparison table grouped by density."""
    if not rows:
        return
    
    # Group by density
    densities = sorted(set(r.density for r in rows))
    agents = sorted(set(r.agent for r in rows))
    
    for n in densities:
        print(f"\n{'='*110}")
        print(f"  DENSITY N = {n}")
        print(f"{'='*110}")
        print(f"{'Agent':<22} {'Seed':<6} {'SafeSuccess%':<14} {'Success%':<10} "
              f"{'Col/Ep':<8} {'AvgLen':<8} {'Jerk':<8} {'MinDist':<8} {'Freeze%':<8}")
        print("-" * 110)
        
        for agent_name in agents:
            agent_rows = [r for r in rows if r.density == n and r.agent == agent_name]
            if not agent_rows:
                continue
            
            for r in agent_rows:
                print(f"{r.agent:<22} {r.seed:<6} "
                      f"{r.safe_success_rate*100:>9.1f}%    "
                      f"{r.success_rate*100:>7.1f}%  "
                      f"{r.collision_per_episode:>6.2f}  "
                      f"{r.avg_episode_length:>6.1f}  "
                      f"{r.avg_jerk:>6.3f}  "
                      f"{r.avg_min_dist:>6.3f}  "
                      f"{r.freezing_rate*100:>5.1f}%")
            
            # Mean across seeds
            if len(agent_rows) > 1:
                mean_safe = np.mean([r.safe_success_rate for r in agent_rows]) * 100
                std_safe = np.std([r.safe_success_rate for r in agent_rows]) * 100
                mean_col = np.mean([r.collision_per_episode for r in agent_rows])
                mean_jerk = np.mean([r.avg_jerk for r in agent_rows])
                mean_len = np.mean([r.avg_episode_length for r in agent_rows])
                mean_min = np.mean([r.avg_min_dist for r in agent_rows])
                
                print(f"  {'└─ MEAN':<20} {'---':<6} "
                      f"{mean_safe:>5.1f}±{std_safe:.1f}%    "
                      f"{'':>9}  "
                      f"{mean_col:>6.2f}  "
                      f"{mean_len:>6.1f}  "
                      f"{mean_jerk:>6.3f}  "
                      f"{mean_min:>6.3f}")
    
    print(f"\n{'='*110}")
    print("KEY: SafeSuccess% = Goal reached AND zero collisions (primary metric)")
    print("     Jerk = Action smoothness (lower = smoother = more socially acceptable)")
    print("     MinDist = Average closest approach to any NPC (higher = safer)")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Evaluation: RL + Analytic agents across density sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full density sweep (random scenario, all agents)
  python eval_unified.py ./runs_social --densities 11,13,15,17,19,21,23 --scenario random

  # Quick comparison at key densities
  python eval_unified.py ./runs_social --densities 11,17,23 -n 50

  # Skip baselines that aren't installed
  python eval_unified.py ./runs_social --no-orca --no-dsrnn --no-lstm-rl --densities 11,17,23

  # Only RL agents (no analytic, no special baselines)
  python eval_unified.py ./runs_social --no-sfm --no-orca --no-sarl --no-dsrnn --no-lstm-rl

  # Evaluate only specific SARL seeds
  python eval_unified.py ./runs_social --sarl-seeds 9,10 --no-sfm --no-orca --densities 11,17,23
        """
    )
    
    parser.add_argument("runs_dir", type=str,
                        help="Path to runs_social directory with trained models")
    parser.add_argument("--densities", type=str, default="6,8,10,12,14",
                        help="Comma-separated NPC densities to evaluate (default: 6,8,10,12,14)")
    parser.add_argument("-n", "--num-episodes", type=int, default=50,
                        help="Episodes per evaluation (default: 50)")
    parser.add_argument("--scenario", type=str, default="corridor",
                        choices=["corridor", "intersection", "circle", "random"],
                        help="Evaluation scenario (default: corridor)")
    parser.add_argument("--max-cycles", type=int, default=100,
                        help="Max steps per episode (default: 100)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: results_{scenario}.csv)")
    parser.add_argument("--no-sfm", action="store_true",
                        help="Skip SFM baseline")
    parser.add_argument("--no-orca", action="store_true",
                        help="Skip ORCA baseline")
    parser.add_argument("--no-sarl", action="store_true",
                        help="Skip SARL baseline")
    parser.add_argument("--no-dsrnn", action="store_true",
                        help="Skip DS-RNN baseline")
    parser.add_argument("--no-lstm-rl", action="store_true",
                        help="Skip LSTM-RL baseline")
    parser.add_argument("--analytic-seeds", type=str, default="42,123,456,9,10",
                        help="Comma-separated seeds for SFM/ORCA env randomness (default: 42,123,456,9,10)")
    parser.add_argument("--sarl-seeds", type=str, default=None,
                        help="Comma-separated SARL seed dirs to evaluate (default: all)")
    parser.add_argument("--dsrnn-seeds", type=str, default=None,
                        help="Comma-separated DS-RNN seed dirs to evaluate (default: all)")
    parser.add_argument("--lstm-rl-seeds", type=str, default=None,
                        help="Comma-separated LSTM_RL seed dirs to evaluate (default: all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-episode details")
    parser.add_argument("--append", action="store_true",
                        help="Append results to existing CSV instead of overwriting")
    
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    densities = [int(x.strip()) for x in args.densities.split(",")]
    analytic_seeds = [int(x.strip()) for x in args.analytic_seeds.split(",")]
    sarl_seeds = [int(x.strip()) for x in args.sarl_seeds.split(",")] if args.sarl_seeds else None
    dsrnn_seeds = [int(x.strip()) for x in args.dsrnn_seeds.split(",")] if args.dsrnn_seeds else None
    lstm_rl_seeds = [int(x.strip()) for x in args.lstm_rl_seeds.split(",")] if args.lstm_rl_seeds else None
    
    if args.output is None:
        args.output = f"results_{args.scenario}.csv"
    
    include_sarl = not args.no_sarl and SARL_AVAILABLE
    include_dsrnn = not args.no_dsrnn and DSRNN_AVAILABLE
    include_lstm_rl = not args.no_lstm_rl and LSTM_RL_AVAILABLE
    
    print(f"\n{'='*70}")
    print(f"  Unified Evaluation")
    print(f"{'='*70}")
    print(f"  Runs dir:   {runs_dir}")
    print(f"  Scenario:   {args.scenario}")
    print(f"  Densities:  {densities}")
    print(f"  Episodes:   {args.num_episodes}")
    print(f"  SFM:        {'✅' if not args.no_sfm else '❌ skipped'}")
    print(f"  ORCA:       {'✅' if (not args.no_orca and RVO2_AVAILABLE) else '❌ skipped'}")
    print(f"  SARL:       {'✅' + (' seeds=' + str(sarl_seeds) if sarl_seeds else ' (all seeds)') if include_sarl else '❌ skipped'}")
    print(f"  DS-RNN:     {'✅' + (' seeds=' + str(dsrnn_seeds) if dsrnn_seeds else ' (all seeds)') if include_dsrnn else '❌ skipped'}")
    print(f"  LSTM-RL:    {'✅' + (' seeds=' + str(lstm_rl_seeds) if lstm_rl_seeds else ' (all seeds)') if include_lstm_rl else '❌ skipped'}")
    print(f"  Analytic seeds: {analytic_seeds}")
    print(f"  Output:     {args.output}")
    
    start_time = time.time()
    
    rows = run_density_sweep(
        runs_dir=runs_dir,
        densities=densities,
        scenario=args.scenario,
        num_episodes=args.num_episodes,
        max_cycles=args.max_cycles,
        include_sfm=not args.no_sfm,
        include_orca=not args.no_orca and RVO2_AVAILABLE,
        include_sarl=include_sarl,
        include_dsrnn=include_dsrnn,
        include_lstm_rl=include_lstm_rl,
        analytic_seeds=analytic_seeds,
        sarl_seeds=sarl_seeds,
        dsrnn_seeds=dsrnn_seeds,
        lstm_rl_seeds=lstm_rl_seeds,
        verbose=args.verbose,
    )
    
    elapsed = time.time() - start_time
    
    # Print results table
    print_results_table(rows)
    
    # Save CSV
    save_results_csv(rows, args.output, append=args.append)
    
    print(f"\n⏱️ Total evaluation time: {elapsed/60:.1f} minutes")
    print(f"📊 Use plot_figures.py to generate figures from {args.output}")


if __name__ == "__main__":
    main()