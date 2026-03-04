#!/usr/bin/env python3
"""
Evaluate DS-RNN models alongside other baselines.

Two usage modes:

1. STANDALONE: Run DS-RNN evaluation independently
   python eval_dsrnn.py ./runs_social/DS_RNN --densities 11,13,15,17,19,21,23 --scenario random

2. PATCH eval_unified.py: Add these lines to eval_unified.py so it auto-handles DS-RNN:

   # At the top (after other imports):
   try:
       from ds_rnn import DSRNNFeaturesExtractor
       DSRNN_AVAILABLE = True
   except ImportError:
       DSRNN_AVAILABLE = False

   # In make_eval_env_with_vecnorm(), change the PPO.load line to:
   custom_objects = {}
   if DSRNN_AVAILABLE:
       custom_objects["DSRNNFeaturesExtractor"] = DSRNNFeaturesExtractor
   model = PPO.load(str(model_file), env=vec_env, custom_objects=custom_objects)

v1.0
"""

import argparse
import sys
import os
import csv
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Register DS-RNN custom class BEFORE loading any models
from ds_rnn import DSRNNFeaturesExtractor, get_dsrnn_policy_kwargs

from env_social_nav import make_social_nav_env, SocialNavConfig, FIXED_OBS_DIM, MAX_NPCS

# Import eval infrastructure from eval_unified
from eval_unified import (
    EvalRow, evaluate_agent, make_eval_env,
    save_results_csv, print_results_table,
)


def load_dsrnn_model(
    model_path: Path,
    num_npcs: int,
    scenario: str,
    max_cycles: int = 100,
) -> Tuple[PPO, VecNormalize]:
    """
    Load DS-RNN model with custom_objects so SB3 finds the extractor class.
    """
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
    
    # Find VecNormalize
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
    
    # CRITICAL: Pass custom_objects so SB3 can reconstruct the DS-RNN extractor
    model = PPO.load(
        str(model_file),
        env=vec_env,
        custom_objects={
            "DSRNNFeaturesExtractor": DSRNNFeaturesExtractor,
        },
    )
    return model, vec_env


def find_dsrnn_seeds(dsrnn_dir: Path) -> List[Path]:
    """Find all seed directories for DS-RNN experiment."""
    seed_dirs = []
    if dsrnn_dir.exists():
        for d in sorted(dsrnn_dir.iterdir()):
            if d.is_dir() and d.name.startswith("seed_"):
                if (d / "best_model.zip").exists() or (d / "final_model.zip").exists():
                    seed_dirs.append(d)
    return seed_dirs


def run_dsrnn_eval(
    dsrnn_dir: Path,
    densities: List[int],
    scenario: str = "random",
    num_episodes: int = 100,
    max_cycles: int = 100,
    agent_name: str = "DS_RNN",
) -> List[EvalRow]:
    """Evaluate DS-RNN across density sweep."""
    
    seed_dirs = find_dsrnn_seeds(dsrnn_dir)
    if not seed_dirs:
        print(f"❌ No DS-RNN seeds found in {dsrnn_dir}")
        return []
    
    print(f"Found DS-RNN seeds: {[d.name for d in seed_dirs]}")
    
    all_rows = []
    
    for n_npcs in densities:
        print(f"\n{'='*70}")
        print(f"  DENSITY N = {n_npcs}")
        print(f"{'='*70}")
        
        for seed_dir in seed_dirs:
            seed = int(seed_dir.name.split("_")[1])
            print(f"  📊 {agent_name} (seed {seed}) @ N={n_npcs}...", end=" ", flush=True)
            
            try:
                model, vec_env = load_dsrnn_model(
                    seed_dir, n_npcs, scenario, max_cycles
                )
                
                row = evaluate_agent(model, vec_env, num_episodes, verbose=False)
                row.agent = agent_name
                row.density = n_npcs
                row.scenario = scenario
                row.seed = seed
                
                all_rows.append(row)
                print(f"SafeSuccess={row.safe_success_rate*100:.0f}% "
                      f"Col={row.collision_per_episode:.2f} "
                      f"Len={row.avg_episode_length:.1f}")
                
                vec_env.close()
            except Exception as e:
                print(f"❌ Error: {e}")
    
    return all_rows


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DS-RNN baseline across density sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate DS-RNN on your density sweep
  python eval_dsrnn.py ./runs_social/DS_RNN --densities 11,13,15,17,19,21,23 --scenario random

  # Quick check
  python eval_dsrnn.py ./runs_social/DS_RNN --densities 11,17,23 -n 20

  # Circle scenario
  python eval_dsrnn.py ./runs_social/DS_RNN --densities 11,13,15,17,19,21,23 --scenario circle

  # Save results to append to existing CSV
  python eval_dsrnn.py ./runs_social/DS_RNN --densities 11,13,15,17,19,21,23 --output dsrnn_random.csv
        """
    )
    
    parser.add_argument("dsrnn_dir", type=str,
                        help="Path to DS-RNN experiment dir (e.g., ./runs_social/DS_RNN)")
    parser.add_argument("--densities", type=str, default="11,13,15,17,19,21,23",
                        help="Comma-separated NPC densities (default: 11,13,15,17,19,21,23)")
    parser.add_argument("-n", "--num-episodes", type=int, default=100,
                        help="Episodes per evaluation (default: 100)")
    parser.add_argument("--scenario", type=str, default="random",
                        choices=["corridor", "intersection", "circle", "random"],
                        help="Evaluation scenario (default: random)")
    parser.add_argument("--max-cycles", type=int, default=100,
                        help="Max steps per episode (default: 100)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: dsrnn_{scenario}.csv)")
    parser.add_argument("--agent-name", type=str, default="DS_RNN",
                        help="Agent name in results (default: DS_RNN)")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing CSV")
    
    args = parser.parse_args()
    
    dsrnn_dir = Path(args.dsrnn_dir)
    densities = [int(x.strip()) for x in args.densities.split(",")]
    
    if args.output is None:
        args.output = f"dsrnn_{args.scenario}.csv"
    
    print(f"\n{'='*70}")
    print(f"  DS-RNN Evaluation")
    print(f"{'='*70}")
    print(f"  Dir:       {dsrnn_dir}")
    print(f"  Scenario:  {args.scenario}")
    print(f"  Densities: {densities}")
    print(f"  Episodes:  {args.num_episodes}")
    print(f"  Output:    {args.output}")
    
    start_time = time.time()
    
    rows = run_dsrnn_eval(
        dsrnn_dir=dsrnn_dir,
        densities=densities,
        scenario=args.scenario,
        num_episodes=args.num_episodes,
        max_cycles=args.max_cycles,
        agent_name=args.agent_name,
    )
    
    elapsed = time.time() - start_time
    
    if rows:
        print_results_table(rows)
        save_results_csv(rows, args.output, append=args.append)
    
    print(f"\n⏱️ Evaluation time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()