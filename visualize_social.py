#!/usr/bin/env python3
"""
visualize_social.py - Watch trained agents navigate through NPCs

Usage:
  # Watch a trained model
  python visualize_social.py ./runs_social/PSS_Social/seed_42
  
  # Compare all experiments side by side (saves video)
  python visualize_social.py ./runs_social --compare --save-video
  
  # Compare with specific seed
  python visualize_social.py ./runs_social --compare --save-video --seed 42
  
  # Compare ALL seeds (creates separate video per seed)
  python visualize_social.py ./runs_social --compare --save-video --all-seeds
  
  # Save only the LAST FRAME as PNG (comparison image)
  python visualize_social.py ./runs_social --compare --save-last-frame --seed 42
  
  # Save last frame for ALL seeds
  python visualize_social.py ./runs_social --compare --save-last-frame --all-seeds
  
  # Custom axis range
  python visualize_social.py ./runs_social --compare --save-video --seed 42 --xlim -10 10 --ylim -10 10
  
  # Adjust settings
  python visualize_social.py ./runs_social/PSS_Social/seed_42 --num-npcs 12 --episodes 5
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import glob
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env_social_nav import make_social_nav_env, SocialNavConfig
from pss_social import get_social_experiment_config, PSSSocialLocalWrapper


def load_model_and_env(run_dir: str, num_npcs: int = 6, scenario: str = "corridor"):
    """Load trained model and create matching environment."""
    
    # Find model - check multiple possible names
    model_path = None
    possible_names = ["model.zip", "final_model.zip", "best_model.zip"]
    
    for name in possible_names:
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        # Try to find any .zip file
        zip_files = glob.glob(os.path.join(run_dir, "*.zip"))
        if zip_files:
            model_path = zip_files[0]
        else:
            raise FileNotFoundError(f"No model found in {run_dir}. Looked for: {possible_names}")
    
    print(f"📂 Loading model: {model_path}")
    
    # Get experiment config
    meta_path = os.path.join(run_dir, "meta.json")
    exp_name = "PSS_Social"  # Default
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        exp_name = meta.get("exp_name", exp_name)
    
    config = get_social_experiment_config(exp_name)
    
    # Create environment
    def _init():
        social_config = SocialNavConfig()
        social_config.num_npcs = num_npcs
        social_config.scenario = scenario
        social_config.max_cycles = 100
        
        env = make_social_nav_env(
            num_npcs=num_npcs,
            scenario=scenario,
            max_cycles=100,
        )
        env = PSSSocialLocalWrapper(env, config)
        return env
    
    env = DummyVecEnv([_init])
    
    # Load VecNormalize if exists
    vecnorm_path = os.path.join(run_dir, "vecnorm.pkl")
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
    
    # Load model
    model = PPO.load(model_path, env=env)
    
    return model, env, exp_name


def extract_positions(env) -> dict:
    """Extract agent positions from environment."""
    try:
        world = env.envs[0].unwrapped.world
        
        ego_pos = world.agents[0].state.p_pos.copy()
        ego_vel = world.agents[0].state.p_vel.copy()
        goal_pos = getattr(world.agents[0], 'goal_pos', np.array([1.2, 0.0]))
        
        npc_positions = []
        npc_goals = []
        for agent in world.agents[1:]:
            npc_positions.append(agent.state.p_pos.copy())
            npc_goals.append(getattr(agent, 'goal_pos', np.zeros(2)))
        
        return {
            'ego_pos': ego_pos,
            'ego_vel': ego_vel,
            'goal_pos': goal_pos,
            'npc_positions': npc_positions,
            'npc_goals': npc_goals,
        }
    except Exception as e:
        return None


def run_episode_with_recording(model, env, max_steps: int = 100):
    """Run one episode and record all positions."""
    obs = env.reset()
    
    trajectory = []
    total_reward = 0
    collisions = 0
    
    for step in range(max_steps):
        # Record positions
        positions = extract_positions(env)
        if positions:
            positions['step'] = step
            trajectory.append(positions)
        
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        
        # Check for collision - use correct key!
        # [FIX] Changed from 'collision' (bool) to 'collisions' (int)
        step_collisions = int(info[0].get('collisions', 0))
        collisions += step_collisions
        
        if done[0]:
            break
    
    return trajectory, total_reward, collisions, info[0].get('goal_reached', False)


def save_frames(trajectory: List[dict], output_dir: str, title: str = "Social Navigation",
                xlim: tuple = (-2.0, 2.0), ylim: tuple = (-2.0, 2.0)):
    """Save each frame as a separate image."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not trajectory:
        print("No trajectory data!")
        return
    
    print(f"💾 Saving {len(trajectory)} frames to {output_dir}/")
    
    for frame_idx, t in enumerate(trajectory):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set up plot with configurable axis limits
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{title} - Frame {frame_idx:03d}")
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        # Draw goal
        goal = trajectory[0]['goal_pos']
        goal_circle = plt.Circle(goal, 0.15, color='green', alpha=0.3, label='Goal')
        ax.add_patch(goal_circle)
        ax.plot(goal[0], goal[1], 'g*', markersize=15)
        
        # Draw ego agent
        ego_circle = plt.Circle(t['ego_pos'], 0.1, color='blue', alpha=0.8, label='Ego (Learning)')
        ax.add_patch(ego_circle)
        
        # Draw ego velocity arrow
        vel = t['ego_vel']
        if np.linalg.norm(vel) > 0.01:
            ax.arrow(t['ego_pos'][0], t['ego_pos'][1], 
                    vel[0]*0.5, vel[1]*0.5,
                    head_width=0.05, head_length=0.02, fc='blue', ec='blue', alpha=0.5)
        
        # Draw NPCs
        for i, npc_pos in enumerate(t['npc_positions']):
            color = 'red'
            circle = plt.Circle(npc_pos, 0.08, color=color, alpha=0.6, 
                              label='NPC' if i == 0 else None)
            ax.add_patch(circle)
        
        # Draw trajectory so far
        traj_x = [trajectory[k]['ego_pos'][0] for k in range(frame_idx+1)]
        traj_y = [trajectory[k]['ego_pos'][1] for k in range(frame_idx+1)]
        ax.plot(traj_x, traj_y, 'b-', alpha=0.3, linewidth=2, label='Trajectory')
        
        # Add info text
        vel_mag = np.linalg.norm(t['ego_vel'])
        dist_to_goal = np.linalg.norm(t['ego_pos'] - goal)
        info_text = f"Step: {frame_idx:3d} | Velocity: {vel_mag:.2f} | Dist to Goal: {dist_to_goal:.2f}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Legend
        ax.legend(loc='upper right')
        
        # Save frame
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        # Progress
        if (frame_idx + 1) % 10 == 0 or frame_idx == len(trajectory) - 1:
            print(f"   Saved frame {frame_idx + 1}/{len(trajectory)}")
    
    print(f"✅ Done! Frames saved to {output_dir}/")
    print(f"   To create video: ffmpeg -framerate 10 -i {output_dir}/frame_%04d.png -c:v libx264 output.mp4")


def visualize_trajectory(trajectory: List[dict], title: str = "Social Navigation",
                         xlim: tuple = (-2.0, 2.0), ylim: tuple = (-2.0, 2.0)):
    """Create animated visualization of trajectory."""
    
    if not trajectory:
        print("No trajectory data to visualize!")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up plot with configurable axis limits
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    
    # Get initial positions
    init = trajectory[0]
    
    # Draw goal
    goal = init['goal_pos']
    goal_circle = plt.Circle(goal, 0.15, color='green', alpha=0.3, label='Goal')
    ax.add_patch(goal_circle)
    ax.plot(goal[0], goal[1], 'g*', markersize=15)
    
    # Draw ego agent
    ego_circle = plt.Circle(init['ego_pos'], 0.1, color='blue', alpha=0.8, label='Ego (Learning)')
    ax.add_patch(ego_circle)
    
    # Draw NPCs
    npc_circles = []
    for i, npc_pos in enumerate(init['npc_positions']):
        circle = plt.Circle(npc_pos, 0.08, color='red', alpha=0.6)
        ax.add_patch(circle)
        npc_circles.append(circle)
    
    # Draw trajectory line
    ego_traj_x = [t['ego_pos'][0] for t in trajectory]
    ego_traj_y = [t['ego_pos'][1] for t in trajectory]
    traj_line, = ax.plot([], [], 'b-', alpha=0.3, linewidth=2)
    
    # Step counter
    step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                        fontsize=12, verticalalignment='top',
                        fontfamily='monospace')
    
    # Legend
    ax.legend(loc='upper right')
    
    def init_anim():
        traj_line.set_data([], [])
        step_text.set_text('')
        return [ego_circle, traj_line, step_text] + npc_circles
    
    def animate(frame):
        if frame >= len(trajectory):
            return [ego_circle, traj_line, step_text] + npc_circles
        
        t = trajectory[frame]
        
        # Update ego position
        ego_circle.center = t['ego_pos']
        
        # Update NPC positions
        for i, npc_pos in enumerate(t['npc_positions']):
            if i < len(npc_circles):
                npc_circles[i].center = npc_pos
        
        # Update trajectory line
        traj_line.set_data(ego_traj_x[:frame+1], ego_traj_y[:frame+1])
        
        # Update step counter
        vel = np.linalg.norm(t['ego_vel'])
        step_text.set_text(f'Step: {frame:3d} | Velocity: {vel:.2f}')
        
        return [ego_circle, traj_line, step_text] + npc_circles
    
    anim = FuncAnimation(fig, animate, init_func=init_anim,
                         frames=len(trajectory), interval=100, blit=True)
    
    return fig, anim


def live_visualization(model, env, num_episodes: int = 3):
    """Run live visualization with matplotlib."""
    
    plt.ion()  # Interactive mode
    
    for ep in range(num_episodes):
        print(f"\n📺 Episode {ep+1}/{num_episodes}")
        
        trajectory, reward, collisions, goal_reached = run_episode_with_recording(model, env)
        
        status = "✅ SUCCESS" if goal_reached else "❌ FAILED"
        print(f"   {status} | Reward: {reward:.1f} | Collisions: {collisions}")
        
        fig, anim = visualize_trajectory(
            trajectory, 
            title=f"Episode {ep+1}: {status} (Reward: {reward:.1f})"
        )
        
        if fig:
            plt.show()
            plt.pause(len(trajectory) * 0.1 + 1)  # Wait for animation
            plt.close(fig)
    
    plt.ioff()


def save_comparison_video(run_dirs: List[str], output_path: str = "comparison.gif",
                          num_npcs: int = 6, scenario: str = "corridor",
                          xlim: tuple = (-2.0, 2.0), ylim: tuple = (-2.0, 2.0)):
    """Create side-by-side comparison video of multiple experiments."""
    
    n_exp = len(run_dirs)
    fig, axes = plt.subplots(1, n_exp, figsize=(5*n_exp, 5))
    if n_exp == 1:
        axes = [axes]
    
    trajectories = []
    exp_names = []
    
    for run_dir in run_dirs:
        try:
            model, env, exp_name = load_model_and_env(run_dir, num_npcs, scenario)
            traj, reward, collisions, goal = run_episode_with_recording(model, env)
            trajectories.append(traj)
            exp_names.append(f"{exp_name}\nR:{reward:.2f} C:{collisions}")
            env.close()
        except Exception as e:
            print(f"Error loading {run_dir}: {e}")
            trajectories.append([])
            exp_names.append("Error")
    
    max_steps = max(len(t) for t in trajectories if t)
    
    # Initialize plots
    ego_circles = []
    npc_circles_list = []
    traj_lines = []
    
    for i, (ax, traj, name) in enumerate(zip(axes, trajectories, exp_names)):
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(name)
        
        if not traj:
            continue
        
        init = traj[0]
        
        # Goal
        goal = init['goal_pos']
        ax.add_patch(plt.Circle(goal, 0.15, color='green', alpha=0.3))
        ax.plot(goal[0], goal[1], 'g*', markersize=15)
        
        # Ego
        ego_circle = plt.Circle(init['ego_pos'], 0.1, color='blue', alpha=0.8)
        ax.add_patch(ego_circle)
        ego_circles.append(ego_circle)
        
        # NPCs
        npc_circles = []
        for npc_pos in init['npc_positions']:
            circle = plt.Circle(npc_pos, 0.08, color='red', alpha=0.6)
            ax.add_patch(circle)
            npc_circles.append(circle)
        npc_circles_list.append(npc_circles)
        
        # Trajectory line
        line, = ax.plot([], [], 'b-', alpha=0.3, linewidth=2)
        traj_lines.append(line)
    
    def animate(frame):
        for i, traj in enumerate(trajectories):
            if not traj or frame >= len(traj):
                continue
            
            t = traj[frame]
            ego_circles[i].center = t['ego_pos']
            
            for j, npc_pos in enumerate(t['npc_positions']):
                if j < len(npc_circles_list[i]):
                    npc_circles_list[i][j].center = npc_pos
            
            xs = [traj[k]['ego_pos'][0] for k in range(frame+1)]
            ys = [traj[k]['ego_pos'][1] for k in range(frame+1)]
            traj_lines[i].set_data(xs, ys)
        
        return ego_circles + sum(npc_circles_list, []) + traj_lines
    
    anim = FuncAnimation(fig, animate, frames=max_steps, interval=100, blit=True)
    
    print(f"💾 Saving to {output_path}...")
    anim.save(output_path, writer=PillowWriter(fps=10))
    print(f"✅ Saved!")
    
    plt.close(fig)


def save_comparison_last_frame(run_dirs: List[str], output_path: str = "comparison_final.png",
                                num_npcs: int = 6, scenario: str = "corridor",
                                xlim: tuple = (-2.0, 2.0), ylim: tuple = (-2.0, 2.0)):
    """Create side-by-side comparison image showing the LAST FRAME of each experiment."""
    
    n_exp = len(run_dirs)
    fig, axes = plt.subplots(1, n_exp, figsize=(6*n_exp, 6))
    if n_exp == 1:
        axes = [axes]
    
    print(f"📸 Generating last frame comparison...")
    
    for i, run_dir in enumerate(run_dirs):
        ax = axes[i]
        
        try:
            model, env, exp_name = load_model_and_env(run_dir, num_npcs, scenario)
            traj, reward, collisions, goal_reached = run_episode_with_recording(model, env)
            env.close()
            
            if not traj:
                ax.set_title(f"{exp_name}\n(No data)")
                continue
            
            # Get LAST frame
            last = traj[-1]
            init = traj[0]
            
            # Status
            status = "✅ SUCCESS" if goal_reached else "❌ FAILED"
            
            # Set up plot
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{exp_name}\n{status} | R:{reward:.2f} | C:{collisions}", fontsize=12)
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            
            # Draw goal
            goal = init['goal_pos']
            ax.add_patch(plt.Circle(goal, 0.15, color='green', alpha=0.3))
            ax.plot(goal[0], goal[1], 'g*', markersize=20, label='Goal')
            
            # Draw start position (hollow circle)
            start = init['ego_pos']
            ax.add_patch(plt.Circle(start, 0.1, fill=False, edgecolor='blue', 
                                    linestyle='--', linewidth=2, alpha=0.5))
            ax.plot(start[0], start[1], 'b+', markersize=10, alpha=0.5)
            
            # Draw full trajectory
            traj_x = [t['ego_pos'][0] for t in traj]
            traj_y = [t['ego_pos'][1] for t in traj]
            ax.plot(traj_x, traj_y, 'b-', alpha=0.4, linewidth=2, label='Trajectory')
            
            # Draw ego at FINAL position
            ego_pos = last['ego_pos']
            ax.add_patch(plt.Circle(ego_pos, 0.1, color='blue', alpha=0.9))
            ax.plot(ego_pos[0], ego_pos[1], 'bo', markersize=12)
            
            # Draw NPCs at FINAL positions
            for j, npc_pos in enumerate(last['npc_positions']):
                color = 'red'
                ax.add_patch(plt.Circle(npc_pos, 0.08, color=color, alpha=0.7))
                if j == 0:
                    ax.plot(npc_pos[0], npc_pos[1], 'ro', markersize=8, label='NPCs')
            
            # Add velocity arrow
            vel = last['ego_vel']
            vel_mag = np.linalg.norm(vel)
            if vel_mag > 0.01:
                ax.arrow(ego_pos[0], ego_pos[1], vel[0]*0.5, vel[1]*0.5,
                        head_width=0.08, head_length=0.04, fc='blue', ec='blue', alpha=0.7)
            
            # Add info text
            dist_to_goal = np.linalg.norm(ego_pos - goal)
            info = f"Steps: {len(traj)} | Dist: {dist_to_goal:.2f} | Vel: {vel_mag:.2f}"
            ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=9,
                   verticalalignment='bottom', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Legend (only for first plot)
            if i == 0:
                ax.legend(loc='upper right', fontsize=8)
                
        except Exception as e:
            ax.set_title(f"Error: {e}")
            print(f"  ❌ Error loading {run_dir}: {e}")
    
    plt.tight_layout()
    
    # Save
    print(f"💾 Saving to {output_path}...")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved!")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize Social Navigation")
    parser.add_argument("path", type=str, help="Run directory or base directory")
    parser.add_argument("--num-npcs", type=int, default=6)
    parser.add_argument("--scenario", type=str, default="corridor")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--compare", action="store_true", help="Compare all experiments")
    parser.add_argument("--save-video", action="store_true", help="Save as GIF")
    parser.add_argument("--save-frames", action="store_true", help="Save as individual PNG images")
    parser.add_argument("--save-last-frame", action="store_true", help="Save only the last frame as PNG")
    parser.add_argument("--output", type=str, default="social_nav.gif")
    parser.add_argument("--xlim", type=float, nargs=2, default=[-2.0, 2.0], 
                        help="X-axis range, e.g. --xlim -10 10")
    parser.add_argument("--ylim", type=float, nargs=2, default=[-2.0, 2.0],
                        help="Y-axis range, e.g. --ylim -10 10")
    parser.add_argument("--seed", type=int, default=None,
                        help="Which seed to use for comparison (default: first available)")
    parser.add_argument("--all-seeds", action="store_true",
                        help="Create separate video/frames for each seed")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎬 SOCIAL NAVIGATION VISUALIZATION")
    print("=" * 60)
    
    if args.compare:
        # Handle --all-seeds: create visualization for each seed
        if args.all_seeds:
            # Find all available seeds
            all_seeds = set()
            for exp in ["Baseline", "Safe_Baseline", "PSS_Social"]:
                pattern = os.path.join(args.path, exp, "seed_*")
                for match in glob.glob(pattern):
                    seed_str = os.path.basename(match).replace("seed_", "")
                    try:
                        all_seeds.add(int(seed_str))
                    except ValueError:
                        pass
            
            if not all_seeds:
                print("❌ No seeds found!")
                return
            
            print(f"Found {len(all_seeds)} seeds: {sorted(all_seeds)}")
            
            xlim = tuple(args.xlim)
            ylim = tuple(args.ylim)
            
            for seed in sorted(all_seeds):
                print(f"\n{'='*60}")
                print(f"🎬 Processing seed {seed}")
                print(f"{'='*60}")
                
                run_dirs = []
                for exp in ["Baseline", "Safe_Baseline", "PSS_Social"]:
                    seed_dir = os.path.join(args.path, exp, f"seed_{seed}")
                    if os.path.exists(seed_dir):
                        run_dirs.append(seed_dir)
                
                if len(run_dirs) < 3:
                    print(f"  ⚠️  Only {len(run_dirs)} experiments found for seed {seed}, skipping")
                    continue
                
                # Generate output filename with seed
                base_output = args.output.replace('.gif', '')
                output_file = f"{base_output}_seed{seed}.gif"
                
                if args.save_frames:
                    for run_dir in run_dirs:
                        model, env, exp_name = load_model_and_env(run_dir, args.num_npcs, args.scenario)
                        traj, reward, col, goal = run_episode_with_recording(model, env)
                        status = "SUCCESS" if goal else "FAILED"
                        output_dir = f"frames_{exp_name}_seed{seed}"
                        save_frames(traj, output_dir, f"{exp_name} (seed {seed}): {status}", xlim=xlim, ylim=ylim)
                        env.close()
                elif args.save_last_frame:
                    base_output = args.output.replace('.gif', '').replace('.png', '')
                    output_file = f"{base_output}_seed{seed}_final.png"
                    save_comparison_last_frame(run_dirs, output_file, args.num_npcs, args.scenario, xlim=xlim, ylim=ylim)
                    print(f"  ✅ Saved: {output_file}")
                elif args.save_video:
                    save_comparison_video(run_dirs, output_file, args.num_npcs, args.scenario, xlim=xlim, ylim=ylim)
                    print(f"  ✅ Saved: {output_file}")
            
            return
        
        # Single seed comparison (original logic)
        run_dirs = []
        for exp in ["Baseline", "Safe_Baseline", "PSS_Social"]:
            if args.seed is not None:
                # Use specific seed
                seed_dir = os.path.join(args.path, exp, f"seed_{args.seed}")
                if os.path.exists(seed_dir):
                    run_dirs.append(seed_dir)
                else:
                    print(f"⚠️  {exp}/seed_{args.seed} not found, skipping")
            else:
                # Use first available seed
                pattern = os.path.join(args.path, exp, "seed_*")
                matches = sorted(glob.glob(pattern))  # Sort for consistency
                if matches:
                    run_dirs.append(matches[0])
        
        if not run_dirs:
            print("❌ No experiment directories found!")
            # List available seeds
            pattern = os.path.join(args.path, "*", "seed_*")
            available = glob.glob(pattern)
            if available:
                print("\nAvailable runs:")
                for run in sorted(available):
                    print(f"  {run}")
            return
        
        print(f"Found {len(run_dirs)} experiments to compare:")
        for rd in run_dirs:
            print(f"  - {rd}")
        
        # Convert args to tuples
        xlim = tuple(args.xlim)
        ylim = tuple(args.ylim)
        
        if args.save_frames:
            # Save frames for each experiment
            for run_dir in run_dirs:
                model, env, exp_name = load_model_and_env(run_dir, args.num_npcs, args.scenario)
                print(f"\n🎬 {exp_name}")
                traj, reward, col, goal = run_episode_with_recording(model, env)
                status = "SUCCESS" if goal else "FAILED"
                print(f"   {status} | Reward: {reward:.1f} | Collisions: {col}")
                
                output_dir = f"frames_{exp_name}"
                save_frames(traj, output_dir, f"{exp_name}: {status}", xlim=xlim, ylim=ylim)
                env.close()
        elif args.save_last_frame:
            # Save only the last frame as a comparison image
            output_file = args.output.replace('.gif', '_final.png')
            if not output_file.endswith('.png'):
                output_file = output_file + '.png'
            save_comparison_last_frame(run_dirs, output_file, args.num_npcs, args.scenario, xlim=xlim, ylim=ylim)
        elif args.save_video:
            save_comparison_video(run_dirs, args.output, args.num_npcs, args.scenario, xlim=xlim, ylim=ylim)
        else:
            # Live comparison (one by one)
            for run_dir in run_dirs:
                model, env, exp_name = load_model_and_env(run_dir, args.num_npcs, args.scenario)
                print(f"\n🎬 {exp_name}")
                live_visualization(model, env, 1)
                env.close()
    else:
        # Single run visualization
        model, env, exp_name = load_model_and_env(args.path, args.num_npcs, args.scenario)
        print(f"Model: {exp_name}")
        print(f"NPCs: {args.num_npcs}, Scenario: {args.scenario}")
        
        # Convert args to tuples
        xlim = tuple(args.xlim)
        ylim = tuple(args.ylim)
        
        if args.save_frames:
            # Save as individual images
            traj, reward, col, goal = run_episode_with_recording(model, env)
            status = "SUCCESS" if goal else "FAILED"
            print(f"Episode: {status} | Reward: {reward:.1f} | Collisions: {col}")
            
            # Create output directory from --output or default
            output_dir = args.output.replace('.gif', '_frames').replace('.png', '_frames')
            if not output_dir.endswith('_frames'):
                output_dir = output_dir + '_frames'
            
            save_frames(traj, output_dir, f"{exp_name}: {status}", xlim=xlim, ylim=ylim)
        
        elif args.save_last_frame:
            # Save only the last frame
            traj, reward, col, goal = run_episode_with_recording(model, env)
            status = "SUCCESS" if goal else "FAILED"
            print(f"Episode: {status} | Reward: {reward:.1f} | Collisions: {col}")
            
            output_file = args.output.replace('.gif', '_final.png')
            if not output_file.endswith('.png'):
                output_file = output_file + '.png'
            
            # Create a single-frame plot
            if traj:
                last = traj[-1]
                init = traj[0]
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.set_xlim(xlim[0], xlim[1])
                ax.set_ylim(ylim[0], ylim[1])
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_title(f"{exp_name}: {status}\nR:{reward:.2f} C:{col}")
                
                # Draw goal
                goal_pos = init['goal_pos']
                ax.add_patch(plt.Circle(goal_pos, 0.15, color='green', alpha=0.3))
                ax.plot(goal_pos[0], goal_pos[1], 'g*', markersize=20, label='Goal')
                
                # Draw trajectory
                traj_x = [t['ego_pos'][0] for t in traj]
                traj_y = [t['ego_pos'][1] for t in traj]
                ax.plot(traj_x, traj_y, 'b-', alpha=0.4, linewidth=2, label='Trajectory')
                
                # Draw ego at final position
                ego_pos = last['ego_pos']
                ax.add_patch(plt.Circle(ego_pos, 0.1, color='blue', alpha=0.9))
                
                # Draw NPCs at final positions
                for j, npc_pos in enumerate(last['npc_positions']):
                    ax.add_patch(plt.Circle(npc_pos, 0.08, color='red', alpha=0.7))
                
                ax.legend(loc='upper right')
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"✅ Saved to {output_file}")
            
        elif args.save_video:
            traj, reward, col, goal = run_episode_with_recording(model, env)
            fig, anim = visualize_trajectory(traj, f"{exp_name}: R={reward:.2f} C={col}", xlim=xlim, ylim=ylim)
            anim.save(args.output, writer=PillowWriter(fps=10))
            print(f"✅ Saved to {args.output}")
        else:
            live_visualization(model, env, args.episodes)
        
        env.close()


if __name__ == "__main__":
    main()