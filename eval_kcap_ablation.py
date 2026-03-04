"""
eval_kcap_ablation.py - 2x2 Factorial Observation Design Ablation

Evaluates ALL agents under 4 observation conditions to isolate the
contribution of K-nearest truncation and distance-based sorting:

  Cell 1 (Kcap+Sort):     k_obs_cap=16,  sort=True   [Full system]
  Cell 2 (NoKcap+Sort):   k_obs_cap=24,  sort=True   [K-cap removed]
  Cell 3 (Kcap+NoSort):   k_obs_cap=16,  sort=False  [Sort removed]
  Cell 4 (NoKcap+NoSort): k_obs_cap=24,  sort=False  [Both removed]

All 4 cells use the SAME trained model + VecNormalize. No retraining.
Degradation is purely from eval-time observation assembly changes.

CLI matches eval_unified.py:
  python eval_kcap_ablation.py ./runs_social \\
      --densities 11,13,15,17,19,21,23 --scenario random -n 100 \\
      --no-sfm --no-orca --output kcap_ablation_random.csv

  # Run only specific conditions (faster)
  python eval_kcap_ablation.py ./runs_social \\
      --densities 11,17,21 -n 50 --conditions baseline,no-kcap

v3.0 - 2x2 factorial, all agents
"""

__version__ = "3.0"

import argparse
import json
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

from env_social_nav import (
    make_social_nav_env, SocialNavConfig,
    FIXED_OBS_DIM, MAX_NPCS,
)

try:
    from policies_analytic import SFMPolicy, ORCAPolicy, RVO2_AVAILABLE
except ImportError:
    RVO2_AVAILABLE = False

try:
    from train_baselines import SARLFeatureExtractor
    SARL_AVAILABLE = True
except ImportError:
    SARL_AVAILABLE = False

try:
    from ds_rnn import DSRNNFeaturesExtractor
    DSRNN_AVAILABLE = True
except ImportError:
    DSRNN_AVAILABLE = False

try:
    from sb3_contrib import RecurrentPPO
    LSTM_RL_AVAILABLE = True
except ImportError:
    LSTM_RL_AVAILABLE = False


# ==============================================================================
# 2x2 Condition Definitions
# ==============================================================================

# Each condition: (label, k_obs_cap, sort_by_distance)
ABLATION_CONDITIONS = {
    "baseline":   ("Kcap+Sort",     16,       True),
    "no-kcap":    ("NoKcap+Sort",   MAX_NPCS, True),
    "no-sort":    ("Kcap+NoSort",   16,       False),
    "no-both":    ("NoKcap+NoSort", MAX_NPCS, False),
}

ALL_CONDITIONS = ["baseline", "no-kcap", "no-sort", "no-both"]


# ==============================================================================
# Data
# ==============================================================================

@dataclass
class EvalRow:
    agent: str = ""
    density: int = 0
    scenario: str = "random"
    seed: int = 0
    condition: str = ""

    success_rate: float = 0.0
    safe_success_rate: float = 0.0
    collision_per_episode: float = 0.0
    avg_episode_length: float = 0.0
    avg_jerk: float = 0.0
    avg_min_dist: float = 0.0
    freezing_rate: float = 0.0

    num_episodes: int = 0


# ==============================================================================
# Evaluation Loops (identical to eval_unified)
# ==============================================================================

def evaluate_agent(agent, vec_env, num_episodes=100, deterministic=True,
                   verbose=False):
    row = EvalRow(num_episodes=num_episodes)

    episode_lengths = []
    episode_collisions_list = []
    episode_min_dists = []

    episodes_done = 0
    obs = vec_env.reset()

    ep_length = 0
    ep_collisions = 0
    ep_min_dist = float('inf')

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

        ep_length += 1
        total_steps += 1

        if prev_action is not None:
            jerk = float(np.mean(np.abs(action[0] - prev_action)))
            total_jerk += jerk
            total_jerk_steps += 1
        prev_action = action[0].copy()

        info_dict = info[0] if isinstance(info, (list, tuple)) else info

        if info_dict.get("collisions", 0) > 0:
            ep_collisions += info_dict["collisions"]
        if info_dict.get("freezing", False):
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

            episode_lengths.append(ep_length)
            episode_collisions_list.append(ep_collisions)
            episode_min_dists.append(
                ep_min_dist if np.isfinite(ep_min_dist) else 0.0
            )

            if verbose and episodes_done % 50 == 0:
                print(f"    [{episodes_done}/{num_episodes}] "
                      f"SafeSR={safe_successes/episodes_done*100:.1f}%")

            ep_length = 0
            ep_collisions = 0
            ep_min_dist = float('inf')
            prev_action = None

    row.success_rate = successes / num_episodes
    row.safe_success_rate = safe_successes / num_episodes
    row.collision_per_episode = (
        np.mean(episode_collisions_list) if episode_collisions_list else 0.0
    )
    row.avg_episode_length = (
        np.mean(episode_lengths) if episode_lengths else 0.0
    )
    row.avg_jerk = total_jerk / max(1, total_jerk_steps)
    row.avg_min_dist = (
        np.mean(episode_min_dists) if episode_min_dists else 0.0
    )
    row.freezing_rate = total_freezes / max(1, total_steps)

    return row


def evaluate_recurrent_agent(model, vec_env, num_episodes=100,
                             deterministic=True):
    assert vec_env.num_envs == 1
    episodes_done = 0
    obs = vec_env.reset()

    lstm_states = None
    episode_starts = np.ones((vec_env.num_envs,), dtype=bool)

    ep_collisions = 0
    ep_min_dist = float('inf')
    ep_length = 0
    prev_action = None

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
            obs, state=lstm_states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        obs, rewards, dones, infos = vec_env.step(action)
        episode_starts = dones

        for i in range(vec_env.num_envs):
            total_steps += 1
            ep_length += 1
            info = (infos[i] if isinstance(infos, (list, tuple, np.ndarray))
                    else infos)

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
    row.collision_per_episode = (
        float(np.mean(all_collisions)) if all_collisions else 0.0
    )
    row.avg_episode_length = (
        float(np.mean(all_lengths)) if all_lengths else 0.0
    )
    row.avg_jerk = total_jerk / max(1, total_jerk_steps)
    row.avg_min_dist = (
        float(np.mean(all_min_dists)) if all_min_dists else 0.0
    )
    row.freezing_rate = total_freezes / max(1, total_steps)
    return row


# ==============================================================================
# Environment + Model Loading
# ==============================================================================

def make_eval_env(num_npcs, scenario, max_cycles,
                  k_obs_cap=16, sort_by_distance=True):
    """Create eval environment with specified observation config."""
    cfg = SocialNavConfig()
    cfg.k_obs_cap = k_obs_cap
    cfg.sort_obs_by_distance = sort_by_distance

    def _make():
        return make_social_nav_env(
            num_npcs=num_npcs,
            scenario=scenario,
            max_cycles=max_cycles,
            config=cfg,
        )
    return DummyVecEnv([_make])


def load_rl_model(model_path, num_npcs, scenario, max_cycles,
                  k_obs_cap, sort_by_distance):
    """Load standard PPO model with ablation-configured env."""
    model_path = Path(model_path)

    model_file = model_path / "best_model.zip"
    if not model_file.exists():
        model_file = model_path / "final_model.zip"
    if not model_file.exists():
        zip_files = list(model_path.glob("*.zip"))
        if zip_files:
            model_file = zip_files[0]
        else:
            raise FileNotFoundError(f"No model in {model_path}")

    vecnorm_file = None
    if "best_model" in model_file.name:
        candidate = model_path / "best_vecnorm.pkl"
        if candidate.exists():
            vecnorm_file = candidate
    if vecnorm_file is None:
        for name in ["vecnorm.pkl", "vecnormalize.pkl"]:
            candidate = model_path / name
            if candidate.exists():
                vecnorm_file = candidate
                break

    vec_env = make_eval_env(num_npcs, scenario, max_cycles,
                            k_obs_cap, sort_by_distance)

    if vecnorm_file is not None and vecnorm_file.exists():
        vec_env = VecNormalize.load(str(vecnorm_file), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(str(model_file), env=vec_env)
    return model, vec_env


def load_baseline_model(seed_dir, num_npcs, scenario, max_cycles,
                        k_obs_cap, sort_by_distance,
                        agent_type="SARL"):
    """Load baseline model (SARL, DS-RNN, LSTM_RL) with ablation env."""
    seed_dir = Path(seed_dir)

    meta_path = seed_dir / "meta.json"
    train_max_cycles = max_cycles
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        train_max_cycles = meta.get("max_cycles", max_cycles)

    model_file = seed_dir / "best_model.zip"
    if not model_file.exists():
        model_file = seed_dir / "final_model.zip"
    if not model_file.exists():
        zip_files = list(seed_dir.glob("*.zip"))
        if zip_files:
            model_file = zip_files[0]
        else:
            raise FileNotFoundError(f"No model in {seed_dir}")

    vecnorm_file = None
    if "best_model" in str(model_file):
        candidate = seed_dir / "best_vecnorm.pkl"
        if candidate.exists():
            vecnorm_file = candidate
    if vecnorm_file is None:
        vecnorm_file = seed_dir / "vecnorm.pkl"
    if not vecnorm_file.exists():
        vecnorm_file = seed_dir / "vecnormalize.pkl"

    vec_env = make_eval_env(num_npcs, scenario, train_max_cycles,
                            k_obs_cap, sort_by_distance)

    if vecnorm_file.exists():
        vec_env = VecNormalize.load(str(vecnorm_file), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    if agent_type == "LSTM_RL":
        if not LSTM_RL_AVAILABLE:
            raise ImportError("sb3-contrib required for LSTM_RL")
        model = RecurrentPPO.load(str(model_file), env=vec_env)
    else:
        custom_objects = {}
        if agent_type == "DS_RNN" and DSRNN_AVAILABLE:
            custom_objects["DSRNNFeaturesExtractor"] = DSRNNFeaturesExtractor
        elif agent_type == "SARL" and SARL_AVAILABLE:
            custom_objects["policy_kwargs"] = dict(
                features_extractor_class=SARLFeatureExtractor,
                features_extractor_kwargs=dict(
                    embed_dim=64, num_heads=4, features_dim=64,
                ),
                net_arch=dict(pi=[64], vf=[64]),
            )
        model = PPO.load(str(model_file), env=vec_env,
                         custom_objects=custom_objects)

    return model, vec_env


# ==============================================================================
# Experiment Discovery (matches eval_unified)
# ==============================================================================

def find_rl_experiments(runs_dir: Path) -> Dict[str, List[Path]]:
    BASELINE_AGENTS = {"SARL", "DS_RNN", "LSTM_RL"}
    experiments = defaultdict(list)
    if not runs_dir.exists():
        return dict(experiments)
    for exp_dir in sorted(runs_dir.iterdir()):
        if exp_dir.is_dir() and exp_dir.name not in BASELINE_AGENTS:
            for seed_dir in sorted(exp_dir.iterdir()):
                if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
                    if ((seed_dir / "best_model.zip").exists() or
                        (seed_dir / "final_model.zip").exists() or
                        list(seed_dir.glob("*.zip"))):
                        experiments[exp_dir.name].append(seed_dir)
    return dict(experiments)


def find_baseline_seeds(runs_dir: Path, agent_name: str,
                        seed_filter: Optional[List[int]] = None) -> List[Path]:
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
# Core: Evaluate one agent under one condition at one density
# ==============================================================================

def eval_one(agent_name, seed_dir, n_npcs, scenario, max_cycles,
             k_obs_cap, sort_by_distance, condition_label,
             num_episodes, agent_type="RL", verbose=False):
    """
    Evaluate a single model under one ablation condition.
    Returns EvalRow or None on failure.
    """
    seed = int(Path(seed_dir).name.split("_")[1])

    try:
        if agent_type in ("SARL", "DS_RNN", "LSTM_RL"):
            model, vec_env = load_baseline_model(
                seed_dir, n_npcs, scenario, max_cycles,
                k_obs_cap, sort_by_distance, agent_type=agent_type,
            )
        else:
            model, vec_env = load_rl_model(
                seed_dir, n_npcs, scenario, max_cycles,
                k_obs_cap, sort_by_distance,
            )

        if agent_type == "LSTM_RL":
            row = evaluate_recurrent_agent(model, vec_env, num_episodes)
        else:
            row = evaluate_agent(model, vec_env, num_episodes,
                                 verbose=verbose)

        row.agent = agent_name
        row.density = n_npcs
        row.scenario = scenario
        row.seed = seed
        row.condition = condition_label

        print(f"SafeSR={row.safe_success_rate*100:.1f}% "
              f"Col={row.collision_per_episode:.2f} "
              f"Freeze={row.freezing_rate*100:.1f}%")

        vec_env.close()
        return row

    except Exception as e:
        print(f"Error: {e}")
        return None


# ==============================================================================
# Main Sweep
# ==============================================================================

def run_2x2_ablation(
    runs_dir,
    densities,
    conditions,
    scenario="random",
    num_episodes=100,
    max_cycles=100,
    include_sfm=False,
    include_orca=False,
    include_sarl=True,
    include_dsrnn=True,
    include_lstm_rl=True,
    analytic_seeds=None,
    sarl_seeds=None,
    dsrnn_seeds=None,
    lstm_rl_seeds=None,
    verbose=False,
):
    if analytic_seeds is None:
        analytic_seeds = [42, 123, 456]

    runs_dir = Path(runs_dir)
    all_rows = []

    # Discover experiments
    rl_experiments = find_rl_experiments(runs_dir)
    if not rl_experiments:
        print(f"No standard RL experiments found in {runs_dir}")
    else:
        print(f"Found RL experiments: {list(rl_experiments.keys())}")

    # Check availability
    if include_orca and not RVO2_AVAILABLE:
        print("ORCA disabled (Python-RVO2 not installed)")
        include_orca = False
    if include_sarl and not SARL_AVAILABLE:
        print("SARL disabled (train_baselines.py not importable)")
        include_sarl = False
    if include_dsrnn and not DSRNN_AVAILABLE:
        print("DS_RNN disabled (ds_rnn.py not importable)")
        include_dsrnn = False
    if include_lstm_rl and not LSTM_RL_AVAILABLE:
        print("LSTM_RL disabled (sb3-contrib not installed)")
        include_lstm_rl = False

    # Resolve conditions
    active_conditions = []
    for cond_key in conditions:
        if cond_key in ABLATION_CONDITIONS:
            label, kcap, sort = ABLATION_CONDITIONS[cond_key]
            active_conditions.append((cond_key, label, kcap, sort))

    print(f"\n2x2 Observation Design Ablation")
    print(f"Conditions: {[c[1] for c in active_conditions]}")
    print(f"Densities:  {densities}")
    print(f"Scenario:   {scenario}")
    print(f"Episodes:   {num_episodes}")
    print()

    for n_npcs in densities:
        ood_tag = " [OOD]" if n_npcs > 16 else ""
        print(f"\n{'='*70}")
        print(f"  DENSITY N = {n_npcs}{ood_tag}")
        print(f"{'='*70}")

        for cond_key, cond_label, k_obs_cap, sort_by_dist in active_conditions:
            print(f"\n  --- Condition: {cond_label} "
                  f"(k_obs_cap={k_obs_cap}, sort={sort_by_dist}) ---")

            # Standard RL agents
            for exp_name, seed_dirs in sorted(rl_experiments.items()):
                for seed_dir in sorted(seed_dirs):
                    seed = int(seed_dir.name.split("_")[1])
                    print(f"    {exp_name} (seed {seed}) ...",
                          end=" ", flush=True)
                    row = eval_one(
                        exp_name, seed_dir, n_npcs, scenario, max_cycles,
                        k_obs_cap, sort_by_dist, cond_label,
                        num_episodes, agent_type="RL", verbose=verbose,
                    )
                    if row:
                        all_rows.append(row)

            # SARL
            if include_sarl:
                for seed_dir in find_baseline_seeds(runs_dir, "SARL",
                                                     sarl_seeds):
                    seed = int(seed_dir.name.split("_")[1])
                    print(f"    SARL (seed {seed}) ...",
                          end=" ", flush=True)
                    row = eval_one(
                        "SARL", seed_dir, n_npcs, scenario, max_cycles,
                        k_obs_cap, sort_by_dist, cond_label,
                        num_episodes, agent_type="SARL", verbose=verbose,
                    )
                    if row:
                        all_rows.append(row)

            # DS_RNN
            if include_dsrnn:
                for seed_dir in find_baseline_seeds(runs_dir, "DS_RNN",
                                                     dsrnn_seeds):
                    seed = int(seed_dir.name.split("_")[1])
                    print(f"    DS_RNN (seed {seed}) ...",
                          end=" ", flush=True)
                    row = eval_one(
                        "DS_RNN", seed_dir, n_npcs, scenario, max_cycles,
                        k_obs_cap, sort_by_dist, cond_label,
                        num_episodes, agent_type="DS_RNN", verbose=verbose,
                    )
                    if row:
                        all_rows.append(row)

            # LSTM_RL
            if include_lstm_rl:
                for seed_dir in find_baseline_seeds(runs_dir, "LSTM_RL",
                                                     lstm_rl_seeds):
                    seed = int(seed_dir.name.split("_")[1])
                    print(f"    LSTM_RL (seed {seed}) ...",
                          end=" ", flush=True)
                    row = eval_one(
                        "LSTM_RL", seed_dir, n_npcs, scenario, max_cycles,
                        k_obs_cap, sort_by_dist, cond_label,
                        num_episodes, agent_type="LSTM_RL", verbose=verbose,
                    )
                    if row:
                        all_rows.append(row)

            # SFM (analytic - doesn't use observations, included for reference)
            if include_sfm:
                for env_seed in analytic_seeds:
                    print(f"    SFM (seed {env_seed}) ...",
                          end=" ", flush=True)
                    try:
                        np.random.seed(env_seed)
                        vec_env = make_eval_env(n_npcs, scenario, max_cycles,
                                                k_obs_cap, sort_by_dist)
                        sfm = SFMPolicy(vec_env)
                        row = evaluate_agent(sfm, vec_env, num_episodes,
                                             verbose=verbose)
                        row.agent = "SFM"
                        row.density = n_npcs
                        row.scenario = scenario
                        row.seed = env_seed
                        row.condition = cond_label
                        all_rows.append(row)
                        print(f"SafeSR={row.safe_success_rate*100:.1f}%")
                        vec_env.close()
                    except Exception as e:
                        print(f"Error: {e}")

            # ORCA
            if include_orca:
                for env_seed in analytic_seeds:
                    print(f"    ORCA (seed {env_seed}) ...",
                          end=" ", flush=True)
                    try:
                        np.random.seed(env_seed)
                        vec_env = make_eval_env(n_npcs, scenario, max_cycles,
                                                k_obs_cap, sort_by_dist)
                        orca = ORCAPolicy(vec_env)
                        row = evaluate_agent(orca, vec_env, num_episodes,
                                             verbose=verbose)
                        row.agent = "ORCA"
                        row.density = n_npcs
                        row.scenario = scenario
                        row.seed = env_seed
                        row.condition = cond_label
                        all_rows.append(row)
                        print(f"SafeSR={row.safe_success_rate*100:.1f}%")
                        vec_env.close()
                    except Exception as e:
                        print(f"Error: {e}")

    return all_rows


# ==============================================================================
# Output
# ==============================================================================

def save_results_csv(rows, output_path, append=False):
    if not rows:
        print("No results to save.")
        return
    field_names = [f.name for f in fields(EvalRow)]
    mode = 'a' if (append and os.path.exists(output_path)) else 'w'
    with open(output_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        if mode == 'w':
            writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    action = "Appended" if mode == 'a' else "Saved"
    print(f"\n{action} {len(rows)} rows to: {output_path}")


def print_results_table(rows):
    """Print 2x2 factorial table: Agent x Condition at each density."""
    all_densities = sorted(set(r.density for r in rows))
    all_conditions = sorted(set(r.condition for r in rows))
    all_agents = sorted(set(r.agent for r in rows))

    # Build lookup: (agent, condition, density) -> [safe_success_rates]
    lookup = defaultdict(list)
    for r in rows:
        lookup[(r.agent, r.condition, r.density)].append(
            r.safe_success_rate * 100
        )

    print(f"\n{'='*80}")
    print(f"  2x2 Observation Design Ablation: Safe Success Rate (%)")
    print(f"{'='*80}")

    for agent in all_agents:
        print(f"\n  {agent}:")
        header = f"    {'Condition':<18}" + "".join(
            f"  N={n:<3}" for n in all_densities
        )
        print(header)
        print(f"    {'-'*len(header)}")

        for cond in all_conditions:
            line = f"    {cond:<18}"
            for n in all_densities:
                vals = lookup.get((agent, cond, n), [])
                if vals:
                    line += f"  {np.mean(vals):>5.1f}"
                else:
                    line += f"  {'--':>5}"
            print(line)

    print()

    # Also print the compact 2x2 table for the paper
    if len(all_conditions) >= 2 and len(all_densities) >= 2:
        print(f"\n{'='*80}")
        print(f"  PAPER TABLE: 2x2 Ablation (mean across seeds)")
        print(f"{'='*80}")
        for agent in all_agents:
            print(f"\n  {agent} | ", end="")
            # Show a representative OOD density (largest)
            ood_n = max(all_densities)
            in_n = min(all_densities)
            print(f"N={in_n} (ID) / N={ood_n} (OOD)")
            print(f"  {'':18}  {'In-Dist':>8}  {'OOD':>8}")
            print(f"  {'-'*40}")
            for cond in all_conditions:
                id_vals = lookup.get((agent, cond, in_n), [])
                ood_vals = lookup.get((agent, cond, ood_n), [])
                id_str = f"{np.mean(id_vals):.1f}" if id_vals else "--"
                ood_str = f"{np.mean(ood_vals):.1f}" if ood_vals else "--"
                print(f"  {cond:<18}  {id_str:>8}  {ood_str:>8}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="2x2 Observation Design Ablation (K-cap x Sort)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full 2x2 ablation, all agents
  python eval_kcap_ablation.py ./runs_social \\
      --densities 11,13,15,17,19,21,23 --scenario random -n 100 \\
      --no-sfm --no-orca --output kcap_ablation_random.csv

  # Only baseline vs no-kcap (faster, 2 cells instead of 4)
  python eval_kcap_ablation.py ./runs_social \\
      --densities 11,17,21 -n 50 --conditions baseline,no-kcap

  # All 4 conditions, quick test
  python eval_kcap_ablation.py ./runs_social \\
      --densities 11,21 -n 20 --no-sfm --no-orca

2x2 conditions:
  baseline  = K-cap ON  + Sort ON   (full system)
  no-kcap   = K-cap OFF + Sort ON   (truncation removed)
  no-sort   = K-cap ON  + Sort OFF  (sorting removed)
  no-both   = K-cap OFF + Sort OFF  (both removed)
        """
    )

    parser.add_argument("runs_dir", type=str,
                        help="Path to runs directory (e.g., ./runs_social)")
    parser.add_argument("--densities", type=str, default="11,13,15,17,19,21",
                        help="Comma-separated NPC densities")
    parser.add_argument("-n", "--num-episodes", type=int, default=100,
                        help="Episodes per seed per density (default: 100)")
    parser.add_argument("--scenario", type=str, default="random",
                        choices=["corridor", "intersection", "circle",
                                 "random"],
                        help="Evaluation scenario (default: random)")
    parser.add_argument("--max-cycles", type=int, default=100,
                        help="Max steps per episode (default: 100)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path")
    parser.add_argument("--conditions", type=str,
                        default="baseline,no-kcap,no-sort,no-both",
                        help="Comma-separated conditions to run "
                        "(default: baseline,no-kcap,no-sort,no-both)")
    parser.add_argument("--no-sfm", action="store_true", help="Skip SFM")
    parser.add_argument("--no-orca", action="store_true", help="Skip ORCA")
    parser.add_argument("--no-sarl", action="store_true", help="Skip SARL")
    parser.add_argument("--no-dsrnn", action="store_true",
                        help="Skip DS-RNN")
    parser.add_argument("--no-lstm-rl", action="store_true",
                        help="Skip LSTM-RL")
    parser.add_argument("--analytic-seeds", type=str, default="42,123,456",
                        help="Seeds for SFM/ORCA")
    parser.add_argument("--sarl-seeds", type=str, default=None,
                        help="SARL seed filter")
    parser.add_argument("--dsrnn-seeds", type=str, default=None,
                        help="DS-RNN seed filter")
    parser.add_argument("--lstm-rl-seeds", type=str, default=None,
                        help="LSTM-RL seed filter")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing CSV")

    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    densities = [int(x.strip()) for x in args.densities.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]
    analytic_seeds = [int(x.strip()) for x in args.analytic_seeds.split(",")]

    # Validate conditions
    for c in conditions:
        if c not in ABLATION_CONDITIONS:
            parser.error(
                f"Unknown condition '{c}'. "
                f"Choose from: {', '.join(ALL_CONDITIONS)}"
            )

    sarl_seeds = None
    if args.sarl_seeds:
        sarl_seeds = [int(x.strip()) for x in args.sarl_seeds.split(",")]
    dsrnn_seeds = None
    if args.dsrnn_seeds:
        dsrnn_seeds = [int(x.strip()) for x in args.dsrnn_seeds.split(",")]
    lstm_rl_seeds = None
    if args.lstm_rl_seeds:
        lstm_rl_seeds = [int(x.strip()) for x in args.lstm_rl_seeds.split(",")]

    if args.output is None:
        args.output = f"kcap_ablation_{args.scenario}.csv"

    include_sarl = not args.no_sarl and SARL_AVAILABLE
    include_dsrnn = not args.no_dsrnn and DSRNN_AVAILABLE
    include_lstm_rl = not args.no_lstm_rl and LSTM_RL_AVAILABLE

    # Pretty print config
    cond_labels = [ABLATION_CONDITIONS[c][0] for c in conditions]
    print(f"\n{'='*60}")
    print(f"  2x2 OBSERVATION DESIGN ABLATION")
    print(f"{'='*60}")
    print(f"  Runs dir:    {runs_dir}")
    print(f"  Scenario:    {args.scenario}")
    print(f"  Densities:   {densities}")
    print(f"  Episodes:    {args.num_episodes}")
    print(f"  Conditions:  {cond_labels}")
    print(f"  SFM:         {'skip' if args.no_sfm else 'include'}")
    print(f"  ORCA:        {'skip' if args.no_orca else 'include'}")
    print(f"  SARL:        {'include' if include_sarl else 'skip'}")
    print(f"  DS_RNN:      {'include' if include_dsrnn else 'skip'}")
    print(f"  LSTM_RL:     {'include' if include_lstm_rl else 'skip'}")
    print(f"  Output:      {args.output}")
    print()

    start_time = time.time()

    rows = run_2x2_ablation(
        runs_dir=runs_dir,
        densities=densities,
        conditions=conditions,
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

    if rows:
        print_results_table(rows)
        save_results_csv(rows, args.output, append=args.append)

    print(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()