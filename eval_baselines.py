#!/usr/bin/env python3
"""
eval_baselines.py - Evaluate SARL and LSTM-RL baselines

Adapts the eval_unified.py pipeline for models that require
special loading (custom feature extractors, RecurrentPPO).

Usage:
  # Evaluate SARL across density sweep
  python eval_baselines.py ./runs_social --agents SARL --scenario random \
    --densities 11,13,15,17,19,21,23 -n 100

  # Evaluate LSTM_RL
  python eval_baselines.py ./runs_social --agents LSTM_RL --scenario random \
    --densities 11,13,15,17,19,21,23 -n 100

  # Evaluate both + existing baselines
  python eval_baselines.py ./runs_social --agents SARL,LSTM_RL --scenario random \
    --densities 11,13,15,17,19,21,23 -n 100

  # Append to existing CSV
  python eval_baselines.py ./runs_social --agents SARL --scenario random \
    --densities 11,13,15,17,19,21,23 -n 100 --append-to merged_random.csv

v1.1 - Baseline evaluation (audit fixes applied)
"""

from __future__ import annotations

__version__ = "1.1"

import os
import sys
import csv
import argparse
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict, fields

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env_social_nav import make_social_nav_env, SocialNavConfig, FIXED_OBS_DIM

# Import SARL feature extractor so PPO.load can deserialize it
from train_baselines import SARLFeatureExtractor

# Import the eval loop from eval_unified
from eval_unified import evaluate_agent, EvalRow

# Try to import RecurrentPPO for LSTM_RL
try:
    from sb3_contrib import RecurrentPPO
    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    SB3_CONTRIB_AVAILABLE = False



# ==============================================================================
# Model Loading
# ==============================================================================

def detect_agent_type(seed_dir: Path) -> str:
    """Detect agent type from metadata."""
    meta_path = seed_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("agent_name", meta.get("exp_name", "unknown"))
    return "unknown"


def load_model(
    seed_dir: Path,
    num_npcs: int,
    scenario: str,
    max_cycles: int = 100,
    use_final: bool = False,
):
    """
    Load a trained model (PPO, SARL, or LSTM_RL) with VecNormalize.

    Reconstructs the eval environment using training config from meta.json
    (audit fix MOD3), overriding only num_npcs, scenario, and disabling
    density randomization for deterministic evaluation.

    Args:
        use_final: If True, prefer final_model.zip + vecnorm.pkl over
                   best_model.zip + best_vecnorm.pkl. Use this when the
                   training callback had a bug and best_model was saved
                   at the wrong step.
    """
    agent_type = detect_agent_type(seed_dir)

    # Load training config from meta.json if available (MOD3)
    meta_path = seed_dir / "meta.json"
    train_max_cycles = max_cycles
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        train_max_cycles = meta.get("max_cycles", max_cycles)

    # Find model file
    if use_final:
        # Prefer final_model (fully trained) over best_model (may be broken)
        model_file = seed_dir / "final_model.zip"
        if not model_file.exists():
            model_file = seed_dir / "best_model.zip"
    else:
        model_file = seed_dir / "best_model.zip"
        if not model_file.exists():
            model_file = seed_dir / "final_model.zip"
    if not model_file.exists():
        zip_files = list(seed_dir.glob("*.zip"))
        if zip_files:
            model_file = zip_files[0]
        else:
            raise FileNotFoundError(f"No model in {seed_dir}")

    # Find VecNormalize — match the model file being loaded
    vecnorm_file = None
    if not use_final and "best_model" in str(model_file):
        candidate = seed_dir / "best_vecnorm.pkl"
        if candidate.exists():
            vecnorm_file = candidate
    if vecnorm_file is None:
        vecnorm_file = seed_dir / "vecnorm.pkl"
    if not vecnorm_file.exists():
        vecnorm_file = seed_dir / "vecnormalize.pkl"

    # Create eval environment (fixed density, no randomization)
    def _make():
        return make_social_nav_env(
            num_npcs=num_npcs,
            scenario=scenario,
            max_cycles=train_max_cycles,
        )

    vec_env = DummyVecEnv([_make])

    # Load VecNormalize stats
    if vecnorm_file.exists():
        vec_env = VecNormalize.load(str(vecnorm_file), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    # Log which checkpoint was loaded
    vn_name = vecnorm_file.name if vecnorm_file.exists() else "NONE"
    print(f"    Loading: {model_file.name} + {vn_name} ({agent_type})", flush=True)

    # Load model based on type
    if agent_type == "LSTM_RL":
        if not SB3_CONTRIB_AVAILABLE:
            raise ImportError("sb3-contrib required to load LSTM_RL models")
        model = RecurrentPPO.load(str(model_file), env=vec_env)
    elif agent_type == "SARL":
        # SARLFeatureExtractor is importable (line 50) so cloudpickle
        # can reconstruct it from the saved policy_kwargs automatically.
        # Do NOT pass custom_objects with policy_kwargs — that overrides
        # the saved hyperparameters and breaks if training config changes.
        model = PPO.load(str(model_file), env=vec_env)
    else:
        # Standard PPO (Baseline, PSS variants, etc.)
        model = PPO.load(str(model_file), env=vec_env)

    return model, vec_env, agent_type


# ==============================================================================
# Evaluation for RecurrentPPO (requires special handling of LSTM states)
# ==============================================================================

def evaluate_recurrent_agent(
    model: "RecurrentPPO",
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
    # Guard: scalar accumulators only work correctly with single env
    assert vec_env.num_envs == 1, (
        f"evaluate_recurrent_agent requires num_envs=1, got {vec_env.num_envs}. "
        f"Per-episode accumulators would mix across environments."
    )
    episodes_done = 0
    obs = vec_env.reset()

    # Initialize LSTM states
    lstm_states = None
    episode_starts = np.ones((vec_env.num_envs,), dtype=bool)

    # Per-episode accumulators (reset each episode)
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

        # Process each env in the VecEnv (typically just 1 for eval)
        for i in range(vec_env.num_envs):
            total_steps += 1
            ep_length += 1
            info = infos[i] if isinstance(infos, (list, tuple, np.ndarray)) else infos

            # ── Per-step jerk (action smoothness) ──
            if prev_action is not None:
                jerk = float(np.mean(np.abs(action[i] - prev_action)))
                total_jerk += jerk
                total_jerk_steps += 1
            prev_action = action[i].copy()

            # ── Per-step collision counting ──
            if info.get("collisions", 0) > 0:
                ep_collisions += info["collisions"]

            # ── Per-step freezing detection ──
            if info.get("freezing", False):
                total_freezes += 1

            # ── Per-step min distance tracking ──
            min_d = info.get("min_dist_to_npc", float('inf'))
            if np.isfinite(min_d):
                ep_min_dist = min(ep_min_dist, min_d)

            # ── Episode end ──
            if dones[i]:
                episodes_done += 1

                # DummyVecEnv: info at done=True IS the terminal episode's info.
                goal = info.get("goal_reached", False)
                # [AUDIT FIX] Use own accumulator ep_collisions consistently.

                if goal:
                    successes += 1
                if goal and ep_collisions == 0:
                    safe_successes += 1

                all_collisions.append(ep_collisions)
                all_lengths.append(ep_length)
                all_min_dists.append(
                    ep_min_dist if np.isfinite(ep_min_dist) else 0.0
                )

                # Reset per-episode accumulators
                ep_collisions = 0
                ep_min_dist = float('inf')
                ep_length = 0
                prev_action = None

                if episodes_done >= num_episodes:
                    break

    row = EvalRow()
    row.success_rate = successes / max(1, episodes_done)
    row.safe_success_rate = safe_successes / max(1, episodes_done)
    row.collision_per_episode = float(np.mean(all_collisions)) if all_collisions else 0.0
    row.avg_episode_length = float(np.mean(all_lengths)) if all_lengths else 0.0
    row.avg_jerk = total_jerk / max(1, total_jerk_steps)
    row.avg_min_dist = float(np.mean(all_min_dists)) if all_min_dists else 0.0
    row.freezing_rate = total_freezes / max(1, total_steps)
    row.num_episodes = episodes_done
    return row


# ==============================================================================
# Experiment Discovery
# ==============================================================================

def find_experiments(runs_dir: Path, agent_filter: List[str] = None) -> Dict[str, List[Path]]:
    """Find experiments matching the agent filter."""
    experiments = defaultdict(list)

    for exp_dir in runs_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        if agent_filter and exp_dir.name not in agent_filter:
            continue
        for seed_dir in exp_dir.iterdir():
            if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
                if (seed_dir / "best_model.zip").exists() or \
                   (seed_dir / "final_model.zip").exists():
                    experiments[exp_dir.name].append(seed_dir)

    return dict(experiments)


# ==============================================================================
# Main Sweep
# ==============================================================================

def run_sweep(
    runs_dir: Path,
    agents: List[str],
    densities: List[int],
    scenario: str,
    num_episodes: int = 100,
    use_final: bool = False,
) -> List[EvalRow]:
    """Run density sweep for specified agents."""
    all_rows = []
    experiments = find_experiments(runs_dir, agents)

    if not experiments:
        print(f"No experiments found for agents {agents} in {runs_dir}")
        return []

    print(f"Found: {list(experiments.keys())}")
    print(f"Densities: {densities}")
    print(f"Scenario: {scenario}")
    print(f"Episodes: {num_episodes}")
    if use_final:
        print(f"Model: final_model.zip + vecnorm.pkl (--use-final)")
    print()

    for n_npcs in densities:
        print(f"\n{'='*60}")
        print(f"  DENSITY N = {n_npcs}")
        print(f"{'='*60}")

        for exp_name, seed_dirs in sorted(experiments.items()):
            for seed_dir in sorted(seed_dirs):
                seed = int(seed_dir.name.split("_")[1])
                print(f"  {exp_name} (seed {seed}) @ N={n_npcs}...", end=" ", flush=True)

                try:
                    model, vec_env, agent_type = load_model(
                        seed_dir, n_npcs, scenario, use_final=use_final,
                    )

                    # Use recurrent eval for LSTM_RL
                    if agent_type == "LSTM_RL":
                        row = evaluate_recurrent_agent(
                            model, vec_env, num_episodes
                        )
                    else:
                        row = evaluate_agent(
                            model, vec_env, num_episodes
                        )

                    row.agent = exp_name
                    row.density = n_npcs
                    row.scenario = scenario
                    row.seed = seed

                    all_rows.append(row)
                    print(
                        f"SafeSR={row.safe_success_rate*100:.0f}% "
                        f"Col={row.collision_per_episode:.2f}"
                    )

                    vec_env.close()
                except Exception as e:
                    print(f"ERROR: {e}")

    return all_rows


def save_csv(rows: List[EvalRow], filepath: str, append: bool = False):
    """Save evaluation rows to CSV."""
    mode = "a" if append else "w"
    write_header = not append or not os.path.exists(filepath)

    field_names = [f.name for f in fields(EvalRow)]

    with open(filepath, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    print(f"\nSaved {len(rows)} rows to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SARL/LSTM-RL baselines")
    parser.add_argument("runs_dir", type=str, help="Path to runs_social directory")
    parser.add_argument("--agents", type=str, default="SARL,LSTM_RL",
                        help="Comma-separated agent names")
    parser.add_argument("--scenario", type=str, default="random")
    parser.add_argument("--densities", type=str, default="11,13,15,17,19,21,23")
    parser.add_argument("-n", "--num-episodes", type=int, default=100)
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output CSV path (default: auto-generated)")
    parser.add_argument("--append-to", type=str, default=None,
                        help="Append results to existing CSV file")
    parser.add_argument("--use-final", action="store_true",
                        help="Use final_model.zip + vecnorm.pkl instead of best_model.zip")

    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    agents = [a.strip() for a in args.agents.split(",")]
    densities = [int(d) for d in args.densities.split(",")]

    rows = run_sweep(runs_dir, agents, densities, args.scenario, args.num_episodes,
                     use_final=args.use_final)

    if not rows:
        print("No results to save.")
        return

    if args.append_to:
        save_csv(rows, args.append_to, append=True)
    else:
        out_path = args.output or f"eval_{'_'.join(agents)}_{args.scenario}.csv"
        save_csv(rows, out_path)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()