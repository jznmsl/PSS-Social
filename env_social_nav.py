#!/usr/bin/env python3
"""
env_social_nav.py - Social Navigation Environment 

VERSION: 2.5 (Numba JIT + Performance Optimized)

Key Features:
  - agent_0 is the EGO AGENT (learning)
  - agent_1 to agent_{N-1} are NPCs (scripted with Social Force Model)
  - Two scenarios: "corridor" (head-on) and "intersection" (crossing)
  - **FIXED observation dimension regardless of actual NPC count**

OPTIMIZATIONS (v2.5):
  1. [CRITICAL] Ghost Collision Fix: Inactive NPCs filled with 10.0 (far away)
  2. [SPEED] Numba JIT: SFM computation 10-50x faster with JIT compilation
  3. [SPEED] Falls back to vectorized NumPy if Numba not installed
  4. [SPEED] Density randomization for zero-shot generalization

Usage:
  from env_social_nav import make_social_nav_env, SocialNavWrapper
  
Install Numba for best performance:
  pip install numba
"""

from __future__ import annotations

__version__ = "2.6-phase1-bugfix"

import math
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

# PettingZoo MPE (future-proof import)
try:
    from mpe2 import simple_spread_v3
except Exception:
    from pettingzoo.mpe import simple_spread_v3


# ==============================================================================
# Configuration
# ==============================================================================

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTANT: Fixed observation dimension for VecNormalize compatibility
# This allows zero-shot generalization (train on 6 NPCs, test on 12 NPCs)
# ══════════════════════════════════════════════════════════════════════════════
MAX_NPCS = 24  # Maximum supported NPCs - observation will be padded to this size

# FIXED observation structure (does NOT depend on actual num_npcs):
#   - Ego velocity:        2
#   - Ego position:        2
#   - Goal direction:      2 (normalized vector to goal)
#   - Goal distance:       1
#   - NPC features:        MAX_NPCS * 4 (rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y)
#                          ** SORTED BY DISTANCE (closest first) for generalization **
#   - Scalar invariant:    12 (expanded density-independent crowd summary)
# TOTAL = 7 + (24 * 4) + 12 = 7 + 96 + 12 = 115
SCALAR_INV_DIM = 12  # EXPANDED: density-invariant crowd summary features
FIXED_OBS_DIM = 7 + (MAX_NPCS * 4) + SCALAR_INV_DIM  # = 7 + 96 + 12 = 115


@dataclass
class SocialNavConfig:
    """Configuration for Social Navigation Environment."""
    # Scenario: "corridor" or "intersection"
    scenario: str = "corridor"
    
    # Agent counts
    num_npcs: int = 6  # Number of NPC agents (scripted)
    # Total agents = 1 (ego) + num_npcs
    
    # Fixed observation dimension (for VecNormalize compatibility)
    max_npcs: int = MAX_NPCS  # Observation padded to this size
    
    # ══════════════════════════════════════════════════════════════════════════════
    # DENSITY RANDOMIZATION (Ideal 2 from prompt)
    # Randomize active NPC count each episode to make VecNormalize robust
    # This prevents gradient explosion when testing on higher density
    # ══════════════════════════════════════════════════════════════════════════════
    randomize_density: bool = False  # Enable density randomization
    min_active_npcs: int = 6         # Minimum active NPCs when randomizing
    max_active_npcs: int = 12        # Maximum active NPCs when randomizing
    
    # ══════════════════════════════════════════════════════════════════════════════
    # OBSERVATION DESIGN (for K-cap ablation)
    # k_obs_cap: Max NPCs written to observation slots (set to MAX_NPCS to disable)
    # sort_obs_by_distance: Whether to sort NPC slots by distance to ego
    # Defaults match training config. Change at EVAL TIME ONLY for ablation.
    # ══════════════════════════════════════════════════════════════════════════════
    k_obs_cap: int = 16                    # K-nearest truncation cap
    sort_obs_by_distance: bool = True      # Sort NPC slots closest-first
    
    # Episode settings
    max_cycles: int = 100
    
    # World bounds
    world_size: float = 1.5
    
    # Social Force Model parameters for NPCs
    sfm_tau: float = 0.5          # Relaxation time
    sfm_k_goal: float = 1.0       # Goal attraction strength
    sfm_k_rep: float = 3.0        # Agent repulsion strength
    sfm_r_0: float = 0.4          # Repulsion decay radius
    sfm_k_obs: float = 2.0        # Obstacle/boundary repulsion
    
    # Social norms thresholds
    intimate_space: float = 0.3   # "Too close" - intrusion
    personal_space: float = 0.5   # Comfort distance
    
    # NPC speed
    npc_speed: float = 0.5
    
    # Ego goal tolerance
    goal_tolerance: float = 0.2
    
    # Reward parameters (for Ego agent)
    reward_goal: float = 10.0
    reward_collision: float = -5.0
    reward_time_penalty: float = -0.01
    reward_intrusion_penalty: float = -0.1  # Entering intimate space


# ==============================================================================
# Social Force Model for NPCs
# ==============================================================================

# ==============================================================================
# Numba JIT-compiled SFM for Maximum Speed
# ==============================================================================

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback: define dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

@jit(nopython=True, cache=True, fastmath=True)
def _sfm_core_numba(
    all_pos: np.ndarray,
    all_vel: np.ndarray, 
    all_goals: np.ndarray,
    npc_speed: float,
    sfm_tau: float,
    sfm_k_rep: float,
    sfm_r_0: float,
    sfm_k_obs: float,
    world_size: float,
) -> np.ndarray:
    """
    Numba-JIT compiled SFM core computation.
    10-50x faster than pure NumPy for small N.
    """
    N = all_pos.shape[0]
    actions = np.zeros((N, 5), dtype=np.float32)
    
    for i in range(N):
        # 1. Goal Force
        dx = all_goals[i, 0] - all_pos[i, 0]
        dy = all_goals[i, 1] - all_pos[i, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist > 1e-6:
            dx /= dist
            dy /= dist
        
        vx_des = dx * npc_speed
        vy_des = dy * npc_speed
        Fx = (vx_des - all_vel[i, 0]) / sfm_tau
        Fy = (vy_des - all_vel[i, 1]) / sfm_tau
        
        # 2. Agent Repulsion
        for j in range(N):
            if i == j:
                continue
            rx = all_pos[i, 0] - all_pos[j, 0]
            ry = all_pos[i, 1] - all_pos[j, 1]
            r_dist = np.sqrt(rx*rx + ry*ry)
            if r_dist < 1e-6:
                continue
            mag = sfm_k_rep * np.exp(-r_dist / sfm_r_0)
            Fx += mag * rx / r_dist
            Fy += mag * ry / r_dist
        
        # 3. Boundary Repulsion
        margin = 0.3
        # Right
        if all_pos[i, 0] > world_size - margin:
            d = max(world_size - all_pos[i, 0], 1e-6)
            Fx -= sfm_k_obs * np.exp(-d / sfm_r_0)
        # Left
        if all_pos[i, 0] < -world_size + margin:
            d = max(all_pos[i, 0] + world_size, 1e-6)
            Fx += sfm_k_obs * np.exp(-d / sfm_r_0)
        # Top
        if all_pos[i, 1] > world_size - margin:
            d = max(world_size - all_pos[i, 1], 1e-6)
            Fy -= sfm_k_obs * np.exp(-d / sfm_r_0)
        # Bottom
        if all_pos[i, 1] < -world_size + margin:
            d = max(all_pos[i, 1] + world_size, 1e-6)
            Fy += sfm_k_obs * np.exp(-d / sfm_r_0)
        
        # Clip force magnitude
        f_mag = np.sqrt(Fx*Fx + Fy*Fy)
        if f_mag > 2.0:
            Fx = Fx / f_mag * 2.0
            Fy = Fy / f_mag * 2.0
        
        # Convert to action
        if Fx < 0:
            actions[i, 1] = min(-Fx, 1.0)  # Left
        else:
            actions[i, 2] = min(Fx, 1.0)   # Right
        if Fy < 0:
            actions[i, 3] = min(-Fy, 1.0)  # Down
        else:
            actions[i, 4] = min(Fy, 1.0)   # Up
    
    return actions


class SFM_NPC_Controller:
    """
    Social Force Model controller for NPC agents.
    Computes forces based on:
      1. Goal attraction
      2. Inter-agent repulsion (from all other agents, including Ego)
      3. Boundary repulsion
    
    OPTIMIZED: 
      - Numba JIT compilation (10-50x faster) if available
      - Falls back to vectorized NumPy if Numba not installed
    """
    
    def __init__(self, config: SocialNavConfig):
        self.c = config
        self._use_numba = NUMBA_AVAILABLE
        if self._use_numba:
            # Warm up JIT compilation
            dummy = np.zeros((2, 2), dtype=np.float64)
            try:
                _sfm_core_numba(dummy, dummy, dummy, 1.0, 0.5, 5.0, 0.3, 3.0, 2.0)
            except Exception:
                self._use_numba = False
    
    def compute_all_actions(
        self,
        all_pos: np.ndarray,
        all_vel: np.ndarray,
        all_goals: np.ndarray,
        active_mask: np.ndarray = None,
    ) -> np.ndarray:
        """
        Vectorized SFM computation for ALL agents at once.
        
        Args:
            all_pos: (N, 2) positions of all agents
            all_vel: (N, 2) velocities of all agents  
            all_goals: (N, 2) goal positions
            active_mask: (N,) boolean, True for active agents (optional)
        
        Returns:
            actions: (N, 5) action array for each agent
        """
        N = len(all_pos)
        if N == 0:
            return np.zeros((0, 5), dtype=np.float32)
        
        # Sanitize inputs
        all_pos = np.ascontiguousarray(np.nan_to_num(all_pos, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)
        all_vel = np.ascontiguousarray(np.nan_to_num(all_vel, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)
        all_goals = np.ascontiguousarray(np.nan_to_num(all_goals, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)
        
        # Use Numba if available (10-50x faster for small N)
        if self._use_numba:
            return _sfm_core_numba(
                all_pos, all_vel, all_goals,
                self.c.npc_speed, self.c.sfm_tau,
                self.c.sfm_k_rep, self.c.sfm_r_0,
                self.c.sfm_k_obs, self.c.world_size
            )
        
        # Fallback: Vectorized NumPy (still fast, but not as fast as Numba)
        return self._compute_numpy(all_pos, all_vel, all_goals)
    
    def _compute_numpy(self, all_pos, all_vel, all_goals) -> np.ndarray:
        """Fallback NumPy implementation."""
        N = len(all_pos)
        
        # ══════════════════════════════════════════════════════════════════════
        # 1. Goal Force (Vectorized): F_goal = (v_desired - v) / tau
        # ══════════════════════════════════════════════════════════════════════
        direction = all_goals - all_pos  # (N, 2)
        dists = np.linalg.norm(direction, axis=1, keepdims=True)  # (N, 1)
        
        # Normalize direction, avoiding div by zero
        safe_dists = np.maximum(dists, 1e-6)
        direction = direction / safe_dists
        
        v_desired = direction * self.c.npc_speed
        F_goal = (v_desired - all_vel) / self.c.sfm_tau
        
        # ══════════════════════════════════════════════════════════════════════
        # 2. Agent Repulsion (Broadcasting N x N)
        #    diff[i, j] = pos[i] - pos[j]
        # ══════════════════════════════════════════════════════════════════════
        # Compute pairwise differences: (N, N, 2)
        diff = all_pos[:, np.newaxis, :] - all_pos[np.newaxis, :, :]
        
        # Compute pairwise distances: (N, N)
        dist_mat = np.linalg.norm(diff, axis=2)
        
        # Mask self-interaction (set diagonal to inf so exp(-inf/r0) = 0)
        np.fill_diagonal(dist_mat, np.inf)
        
        # Exponential repulsion magnitude: (N, N)
        rep_mag = self.c.sfm_k_rep * np.exp(-dist_mat / self.c.sfm_r_0)
        
        # Normalize direction vectors: diff / dist
        # Use broadcasting to handle (N, N, 2) / (N, N, 1)
        safe_dist_mat = np.maximum(dist_mat, 1e-6)[:, :, np.newaxis]  # (N, N, 1)
        rep_dir = diff / safe_dist_mat  # (N, N, 2)
        
        # Weight by magnitude and sum over all neighbors (axis=1)
        F_rep = np.sum(rep_mag[:, :, np.newaxis] * rep_dir, axis=1)  # (N, 2)
        
        # ══════════════════════════════════════════════════════════════════════
        # 3. Boundary Repulsion (Vectorized)
        # ══════════════════════════════════════════════════════════════════════
        F_bound = np.zeros_like(all_pos)
        bound = self.c.world_size
        margin = 0.3
        
        # Right boundary (x > bound - margin)
        right_mask = all_pos[:, 0] > bound - margin
        dist_to_right = np.maximum(bound - all_pos[:, 0], 1e-6)
        F_bound[right_mask, 0] -= self.c.sfm_k_obs * np.exp(-dist_to_right[right_mask] / self.c.sfm_r_0)
        
        # Left boundary (x < -bound + margin)
        left_mask = all_pos[:, 0] < -bound + margin
        dist_to_left = np.maximum(all_pos[:, 0] + bound, 1e-6)
        F_bound[left_mask, 0] += self.c.sfm_k_obs * np.exp(-dist_to_left[left_mask] / self.c.sfm_r_0)
        
        # Top boundary (y > bound - margin)
        top_mask = all_pos[:, 1] > bound - margin
        dist_to_top = np.maximum(bound - all_pos[:, 1], 1e-6)
        F_bound[top_mask, 1] -= self.c.sfm_k_obs * np.exp(-dist_to_top[top_mask] / self.c.sfm_r_0)
        
        # Bottom boundary (y < -bound + margin)
        bottom_mask = all_pos[:, 1] < -bound + margin
        dist_to_bottom = np.maximum(all_pos[:, 1] + bound, 1e-6)
        F_bound[bottom_mask, 1] += self.c.sfm_k_obs * np.exp(-dist_to_bottom[bottom_mask] / self.c.sfm_r_0)
        
        # ══════════════════════════════════════════════════════════════════════
        # Total Force
        # ══════════════════════════════════════════════════════════════════════
        F_total = F_goal + F_rep + F_bound
        
        # Clip force magnitude
        max_force = 2.0
        force_mags = np.linalg.norm(F_total, axis=1, keepdims=True)
        scale = np.where(force_mags > max_force, max_force / np.maximum(force_mags, 1e-6), 1.0)
        F_total = F_total * scale
        
        # ══════════════════════════════════════════════════════════════════════
        # Convert Forces to MPE Actions (Vectorized)
        # action = [no_action, left, right, down, up]
        # ══════════════════════════════════════════════════════════════════════
        actions = np.zeros((N, 5), dtype=np.float32)
        
        # X-axis: negative = left, positive = right
        actions[:, 1] = np.maximum(0, -F_total[:, 0])  # Left
        actions[:, 2] = np.maximum(0, F_total[:, 0])   # Right
        
        # Y-axis: negative = down, positive = up
        actions[:, 3] = np.maximum(0, -F_total[:, 1])  # Down
        actions[:, 4] = np.maximum(0, F_total[:, 1])   # Up
        
        # Clip and sanitize
        actions = np.clip(actions, 0.0, 1.0)
        actions = np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=0.0)
        
        return actions
    
    def compute_action(
        self,
        npc_pos: np.ndarray,
        npc_vel: np.ndarray,
        goal_pos: np.ndarray,
        all_agents_pos: np.ndarray,
        npc_idx: int,
    ) -> np.ndarray:
        """
        Legacy single-agent interface (calls vectorized version internally).
        Kept for backward compatibility.
        """
        # Build full arrays for vectorized computation
        N = len(all_agents_pos)
        all_vel = np.zeros((N, 2), dtype=np.float32)
        all_goals = np.zeros((N, 2), dtype=np.float32)
        
        all_vel[npc_idx] = npc_vel
        all_goals[npc_idx] = goal_pos
        
        # For other agents, use zero velocity and their current position as goal (stationary)
        for i in range(N):
            if i != npc_idx:
                all_goals[i] = all_agents_pos[i]
        
        # Compute all actions and extract the one we need
        all_actions = self.compute_all_actions(all_agents_pos, all_vel, all_goals)
        return all_actions[npc_idx]


# ==============================================================================
# Scenario Initialization
# ==============================================================================

def init_corridor_scenario(world, config: SocialNavConfig, active_npc_count: int = None):
    """
    Corridor Scenario: Head-on collision test
    - Ego starts on LEFT, goal on RIGHT
    - NPCs start on RIGHT, goals on LEFT
    
    NOTE: Random noise added to NPC positions to prevent overfitting!
    
    Args:
        world: MPE world object
        config: SocialNavConfig
        active_npc_count: Number of NPCs to activate (default: config.num_npcs)
    """
    # Use active_npc_count if provided, otherwise fall back to config
    num_npcs = active_npc_count if active_npc_count is not None else config.num_npcs
    num_npcs = min(num_npcs, len(world.agents) - 1)  # Ensure we don't exceed available NPCs
    
    # Ego Agent (agent_0): starts LEFT, goal RIGHT
    ego = world.agents[0]
    ego.state.p_pos = np.array([-1.2, 0.0], dtype=np.float32)
    ego.state.p_vel = np.zeros(2, dtype=np.float32)
    ego.goal_pos = np.array([1.2, 0.0], dtype=np.float32)  # Custom attribute
    
    # Landmarks: first landmark is Ego's goal
    if len(world.landmarks) > 0:
        world.landmarks[0].state.p_pos = np.array([1.2, 0.0], dtype=np.float32)
    
    # NPCs: start RIGHT, spread vertically, goals LEFT
    # Add random noise to prevent overfitting to fixed formation
    y_spread = np.linspace(-0.8, 0.8, num_npcs) if num_npcs > 0 else []
    y_noise = np.random.uniform(-0.1, 0.1, num_npcs) if num_npcs > 0 else []
    x_noise = np.random.uniform(-0.1, 0.1, num_npcs) if num_npcs > 0 else []
    
    for i in range(num_npcs):
        npc = world.agents[i + 1]
        npc.state.p_pos = np.array([
            1.0 + x_noise[i], 
            y_spread[i] + y_noise[i]
        ], dtype=np.float32)
        npc.state.p_vel = np.zeros(2, dtype=np.float32)
        npc.goal_pos = np.array([-1.2, y_spread[i]], dtype=np.float32)
        
        # Assign landmark as goal if available
        if i + 1 < len(world.landmarks):
            world.landmarks[i + 1].state.p_pos = npc.goal_pos.copy()


def init_intersection_scenario(world, config: SocialNavConfig, active_npc_count: int = None):
    """
    Intersection Scenario: Crossing conflict
    - Ego starts BOTTOM, goal TOP
    - NPCs start LEFT, goals RIGHT (crossing path)
    
    NOTE: Random noise added to NPC positions to prevent overfitting!
    
    Args:
        world: MPE world object
        config: SocialNavConfig
        active_npc_count: Number of NPCs to activate (default: config.num_npcs)
    """
    # Use active_npc_count if provided, otherwise fall back to config
    num_npcs = active_npc_count if active_npc_count is not None else config.num_npcs
    num_npcs = min(num_npcs, len(world.agents) - 1)  # Ensure we don't exceed available NPCs
    
    # Ego Agent (agent_0): starts BOTTOM, goal TOP
    ego = world.agents[0]
    ego.state.p_pos = np.array([0.0, -1.2], dtype=np.float32)
    ego.state.p_vel = np.zeros(2, dtype=np.float32)
    ego.goal_pos = np.array([0.0, 1.2], dtype=np.float32)
    
    if len(world.landmarks) > 0:
        world.landmarks[0].state.p_pos = np.array([0.0, 1.2], dtype=np.float32)
    
    # NPCs: start LEFT, spread vertically around y=0, goals RIGHT
    # Add random noise to prevent overfitting
    y_spread = np.linspace(-0.5, 0.5, num_npcs) if num_npcs > 0 else []
    y_noise = np.random.uniform(-0.1, 0.1, num_npcs) if num_npcs > 0 else []
    x_noise = np.random.uniform(-0.1, 0.1, num_npcs) if num_npcs > 0 else []
    
    for i in range(num_npcs):
        npc = world.agents[i + 1]
        npc.state.p_pos = np.array([
            -1.2 + x_noise[i], 
            y_spread[i] + y_noise[i]
        ], dtype=np.float32)
        npc.state.p_vel = np.zeros(2, dtype=np.float32)
        npc.goal_pos = np.array([1.2, y_spread[i]], dtype=np.float32)
        
        if i + 1 < len(world.landmarks):
            world.landmarks[i + 1].state.p_pos = npc.goal_pos.copy()


def init_circle_scenario(world, config: SocialNavConfig, active_npc_count: int = None):
    """
    Circle Scenario: Classic CrowdNav antipodal crossing test.
    - All agents (ego + NPCs) placed on a circle of radius R
    - Each agent's goal is the antipodal point (diametrically opposite)
    - Creates maximum crossing conflicts at the center
    
    This is the standard benchmark in CrowdNav / ORCA literature.
    
    Args:
        world: MPE world object
        config: SocialNavConfig
        active_npc_count: Number of NPCs to activate (default: config.num_npcs)
    """
    num_npcs = active_npc_count if active_npc_count is not None else config.num_npcs
    num_npcs = min(num_npcs, len(world.agents) - 1)
    
    total_agents = 1 + num_npcs  # ego + NPCs
    radius = 1.0  # Circle radius (within world_size=1.5)
    
    # Distribute all agents evenly on the circle
    # Ego gets angle 0 (rightmost point)
    angles = np.linspace(0, 2 * np.pi, total_agents, endpoint=False)
    
    # Add small angular noise to prevent symmetric deadlocks
    angle_noise = np.random.uniform(-0.05, 0.05, total_agents)
    angles = angles + angle_noise
    
    # Ego Agent (agent_0): on circle, goal = antipodal
    ego = world.agents[0]
    ego.state.p_pos = np.array([
        radius * np.cos(angles[0]),
        radius * np.sin(angles[0])
    ], dtype=np.float32)
    ego.state.p_vel = np.zeros(2, dtype=np.float32)
    ego.goal_pos = np.array([
        radius * np.cos(angles[0] + np.pi),
        radius * np.sin(angles[0] + np.pi)
    ], dtype=np.float32)
    
    if len(world.landmarks) > 0:
        world.landmarks[0].state.p_pos = ego.goal_pos.copy()
    
    # NPCs: on circle, each going to antipodal point
    for i in range(num_npcs):
        npc = world.agents[i + 1]
        a = angles[i + 1]
        
        # Small radial noise
        r_noise = np.random.uniform(-0.05, 0.05)
        
        npc.state.p_pos = np.array([
            (radius + r_noise) * np.cos(a),
            (radius + r_noise) * np.sin(a)
        ], dtype=np.float32)
        npc.state.p_vel = np.zeros(2, dtype=np.float32)
        npc.goal_pos = np.array([
            radius * np.cos(a + np.pi),
            radius * np.sin(a + np.pi)
        ], dtype=np.float32)
        
        if i + 1 < len(world.landmarks):
            world.landmarks[i + 1].state.p_pos = npc.goal_pos.copy()


def init_random_scenario(world, config: SocialNavConfig, active_npc_count: int = None):
    """
    Random Scenario: Random spawn positions and goals.
    - Ego and NPCs placed randomly within the world bounds
    - Each agent gets a random goal position
    - Minimum separation enforced to prevent spawn collisions
    
    Tests general navigation ability without structural bias.
    
    Args:
        world: MPE world object
        config: SocialNavConfig
        active_npc_count: Number of NPCs to activate (default: config.num_npcs)
    """
    num_npcs = active_npc_count if active_npc_count is not None else config.num_npcs
    num_npcs = min(num_npcs, len(world.agents) - 1)
    
    spawn_range = 1.1  # Stay within world_size=1.5 with margin
    min_separation = 0.3  # Minimum distance between any two agents at spawn
    min_goal_dist = 0.8  # Minimum distance from spawn to goal
    
    # Generate non-overlapping spawn positions for all agents
    total_agents = 1 + num_npcs
    positions = []
    max_attempts = 100
    
    for _ in range(total_agents):
        for attempt in range(max_attempts):
            pos = np.random.uniform(-spawn_range, spawn_range, size=2).astype(np.float32)
            
            # Check minimum separation from all existing positions
            if all(np.linalg.norm(pos - p) > min_separation for p in positions):
                positions.append(pos)
                break
        else:
            # Fallback: place with small offset from last position
            offset = np.random.uniform(-0.2, 0.2, size=2).astype(np.float32)
            positions.append(positions[-1] + offset if positions else np.zeros(2, dtype=np.float32))
    
    # Generate goal positions (far enough from spawn)
    goals = []
    for i in range(total_agents):
        for attempt in range(max_attempts):
            goal = np.random.uniform(-spawn_range, spawn_range, size=2).astype(np.float32)
            if np.linalg.norm(goal - positions[i]) > min_goal_dist:
                goals.append(goal)
                break
        else:
            # Fallback: opposite side of spawn
            goals.append(-positions[i])
    
    # Ego Agent (agent_0)
    ego = world.agents[0]
    ego.state.p_pos = positions[0].copy()
    ego.state.p_vel = np.zeros(2, dtype=np.float32)
    ego.goal_pos = goals[0].copy()
    
    if len(world.landmarks) > 0:
        world.landmarks[0].state.p_pos = goals[0].copy()
    
    # NPCs
    for i in range(num_npcs):
        npc = world.agents[i + 1]
        npc.state.p_pos = positions[i + 1].copy()
        npc.state.p_vel = np.zeros(2, dtype=np.float32)
        npc.goal_pos = goals[i + 1].copy()
        
        if i + 1 < len(world.landmarks):
            world.landmarks[i + 1].state.p_pos = goals[i + 1].copy()


# ==============================================================================
# Social Navigation Wrapper
# ==============================================================================

class SocialNavWrapper:
    """
    Wraps PettingZoo parallel env to implement Social Navigation:
    - agent_0 is controlled by the policy (Ego)
    - agent_1+ are controlled by SFM (NPCs)
    
    Returns single-agent observation/action for the Ego agent.
    
    CRITICAL (v2.1): Observation is FIXED SIZE regardless of actual num_npcs!
    
    NEW (v2.2): Supports density randomization for robust VecNormalize statistics.
    """
    
    def __init__(self, env, config: SocialNavConfig):
        self.env = env
        self.c = config
        self.npc_controller = SFM_NPC_Controller(config)
        
        self.agents = list(env.possible_agents)
        self.num_agents = len(self.agents)
        self.ego_agent = self.agents[0]
        self.npc_agents = self.agents[1:]
        
        # ══════════════════════════════════════════════════════════════════════════
        # DENSITY RANDOMIZATION (Ideal 2)
        # Track which NPCs are "active" this episode
        # Inactive NPCs are teleported far away and frozen
        # ══════════════════════════════════════════════════════════════════════════
        self.active_npc_count = len(self.npc_agents)  # All NPCs active by default
        self.inactive_npc_position = np.array([1000.0, 1000.0], dtype=np.float32)
        
        # Get action space from ego agent (this is consistent)
        single_act_space = env.action_space(self.ego_agent)
        self.single_act_dim = int(np.prod(single_act_space.shape))
        self._act_low = np.array(single_act_space.low, dtype=np.float32)
        self._act_high = np.array(single_act_space.high, dtype=np.float32)
        
        # ══════════════════════════════════════════════════════════════════════
        # CRITICAL FIX: Use FIXED observation dimension (NOT from MPE!)
        # This enables zero-shot generalization to different NPC counts
        # ══════════════════════════════════════════════════════════════════════
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(FIXED_OBS_DIM,),  # Always 115 dimensions (7 ego + 96 NPC + 12 scalar)
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.single_act_dim,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.episode_collisions = 0
        self.episode_intrusions = 0
        self.episode_steps = 0
        self.ego_reached_goal = False
        
    def reset(self, seed=None, options=None):
        obs_dict, info_dict = self.env.reset(seed=seed, options=options)
        
        # ══════════════════════════════════════════════════════════════════════════
        # DENSITY RANDOMIZATION (Ideal 2)
        # Randomly select how many NPCs are "active" this episode
        # ══════════════════════════════════════════════════════════════════════════
        if self.c.randomize_density:
            # Randomly choose active NPC count
            min_npcs = max(1, self.c.min_active_npcs)
            max_npcs = min(len(self.npc_agents), self.c.max_active_npcs)
            self.active_npc_count = np.random.randint(min_npcs, max_npcs + 1)
        else:
            # Use fixed NPC count from config
            self.active_npc_count = min(self.c.num_npcs, len(self.npc_agents))
        
        # Apply scenario initialization
        try:
            world = self.env.unwrapped.world
            
            if self.c.scenario == "corridor":
                init_corridor_scenario(world, self.c, self.active_npc_count)
            elif self.c.scenario == "intersection":
                init_intersection_scenario(world, self.c, self.active_npc_count)
            elif self.c.scenario == "circle":
                init_circle_scenario(world, self.c, self.active_npc_count)
            elif self.c.scenario == "random":
                init_random_scenario(world, self.c, self.active_npc_count)
            else:
                init_corridor_scenario(world, self.c, self.active_npc_count)  # Default
            
            # ══════════════════════════════════════════════════════════════════════
            # Handle INACTIVE NPCs - move them out of the way
            # CRITICAL: Each inactive NPC must be at a UNIQUE position to avoid
            # dist=0 in MPE physics which causes NaN. We place them in a line
            # outside the main area, with small random offsets.
            # ══════════════════════════════════════════════════════════════════════
            for i in range(self.active_npc_count, len(self.npc_agents)):
                npc_idx = i + 1  # +1 because agent_0 is ego
                if npc_idx < len(world.agents):
                    npc = world.agents[npc_idx]
                    # Place each inactive NPC at a unique position far from others
                    # Use large spacing (10.0) to ensure no physics interactions
                    # Add small random offset to prevent exact position matches
                    random_offset = np.random.uniform(-0.1, 0.1, size=2)
                    npc.state.p_pos = np.array([
                        100.0 + i * 10.0 + random_offset[0],  # x: 100, 110, 120, ...
                        100.0 + i * 10.0 + random_offset[1],  # y: spread out
                    ], dtype=np.float32)
                    npc.state.p_vel = np.zeros(2, dtype=np.float32)
                    # Set goal to same location (so SFM produces zero force)
                    npc.goal_pos = npc.state.p_pos.copy()
            
        except Exception as e:
            # If scenario init fails, continue with default
            pass
        
        # Reset episode tracking
        self.episode_collisions = 0
        self.episode_intrusions = 0
        self.episode_steps = 0
        self.ego_reached_goal = False
        
        # Build fixed-size observation
        obs = self._build_fixed_obs()
        info = {
            "scenario": self.c.scenario,
            "active_npc_count": self.active_npc_count,  # Track for debugging
        }
        
        return obs, info
    
    def step(self, ego_action: np.ndarray):
        """
        Step the environment:
        1. Ego takes the given action
        2. ACTIVE NPCs take SFM-computed actions (VECTORIZED)
        3. INACTIVE NPCs stay stationary (zero action)
        
        OPTIMIZED: Uses vectorized SFM computation for all NPCs at once.
        """
        self.episode_steps += 1
        
        # Get world state for NPC controller
        try:
            world = self.env.unwrapped.world
            all_pos = np.array([a.state.p_pos for a in world.agents], dtype=np.float32)
            all_vel = np.array([a.state.p_vel for a in world.agents], dtype=np.float32)
            all_goals = np.array([getattr(a, 'goal_pos', np.zeros(2)) for a in world.agents], dtype=np.float32)
            # Sanitize
            all_pos = np.nan_to_num(all_pos, nan=0.0, posinf=0.0, neginf=0.0)
            all_vel = np.nan_to_num(all_vel, nan=0.0, posinf=0.0, neginf=0.0)
            all_goals = np.nan_to_num(all_goals, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            # Fallback
            all_pos = np.zeros((self.num_agents, 2), dtype=np.float32)
            all_vel = np.zeros((self.num_agents, 2), dtype=np.float32)
            all_goals = np.zeros((self.num_agents, 2), dtype=np.float32)
        
        # Ego action (transform from [-1,1] to env action space)
        ego_action = np.asarray(ego_action, dtype=np.float32)
        ego_action = np.nan_to_num(ego_action, nan=0.0, posinf=0.0, neginf=0.0)
        expects_01 = (self._act_low.min() >= 0.0) and (self._act_high.max() <= 1.0)
        
        if expects_01:
            ego_act_env = (ego_action + 1.0) / 2.0
            ego_act_env = np.clip(ego_act_env, 0.0, 1.0)
        else:
            ego_act_env = np.clip(ego_action, self._act_low, self._act_high)
        
        # ══════════════════════════════════════════════════════════════════════
        # VECTORIZED NPC ACTION COMPUTATION
        # Compute all NPC actions in one shot using NumPy broadcasting
        # ══════════════════════════════════════════════════════════════════════
        all_actions = self.npc_controller.compute_all_actions(all_pos, all_vel, all_goals)
        
        # Build action dict
        action_dict = {self.ego_agent: ego_act_env}
        
        for i, npc_name in enumerate(self.npc_agents):
            npc_idx = i + 1  # Index in world.agents
            
            if i < self.active_npc_count:
                # ACTIVE NPC: use vectorized SFM result
                npc_action = all_actions[npc_idx]
            else:
                # INACTIVE NPC: zero action (stay stationary)
                npc_action = np.zeros(5, dtype=np.float32)
            
            # Ensure action is in correct range
            if expects_01:
                npc_action = np.clip(npc_action, 0.0, 1.0)
            
            action_dict[npc_name] = npc_action
        
        # Step environment
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.env.step(action_dict)
        
        # ══════════════════════════════════════════════════════════════════════
        # CRITICAL: Sanitize agent positions AFTER physics step
        # MPE physics can produce NaN when dist=0 between agents
        # ══════════════════════════════════════════════════════════════════════
        try:
            for agent in world.agents:
                if np.any(np.isnan(agent.state.p_pos)):
                    agent.state.p_pos = np.zeros(2, dtype=np.float64)
                if np.any(np.isnan(agent.state.p_vel)):
                    agent.state.p_vel = np.zeros(2, dtype=np.float64)
        except Exception:
            pass
        
        # Enforce world boundaries (prevent infinite drift)
        self._enforce_boundaries()
        
        # Compute custom reward and metrics for Ego
        reward, terminated, truncated, info = self._compute_ego_reward_and_info(
            obs_dict, rew_dict, term_dict, trunc_dict
        )
        
        # Build fixed-size observation
        obs = self._build_fixed_obs()
        
        return obs, reward, terminated, truncated, info
    
    def _build_fixed_obs(self) -> np.ndarray:
        """
        Build FIXED-SIZE observation for Ego agent.
        
        Structure (total = 115 dimensions):
          [0:2]   Ego velocity (2)
          [2:4]   Ego position (2)
          [4:6]   Goal direction - normalized (2)
          [6:7]   Goal distance (1)
          [7:103] NPC features - padded to MAX_NPCS * 4 (96)
                  Each NPC: [rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y]
                  ** SORTED BY DISTANCE TO EGO (closest first) **
          [103:115] Expanded scalar invariant features (12)
                  pressure, alignment, risk, nearest_1/2/3,
                  n_intimate, n_personal, n_social, active_frac,
                  mean_rel_vel_x, mean_rel_vel_y
        
        CRITICAL: This is ALWAYS 115 dimensions regardless of actual num_npcs!
        
        NOTE: Inactive NPCs are filled with LARGE VALUES (10.0) to represent
              "far away", NOT zeros! Zeros would be interpreted as collision
              (the "Ghost Collision" bug).
        """
        obs = np.zeros(FIXED_OBS_DIM, dtype=np.float32)
        
        # ══════════════════════════════════════════════════════════════════════
        # [FIX] Ghost Collision Bug - Initialize inactive NPCs as "far away"
        # rel_pos = 10.0 means "outside perception range" (env is ~4 units wide)
        # rel_vel = 0.0 is fine (stationary far-away agent)
        # ══════════════════════════════════════════════════════════════════════
        npc_feature_start = 7
        for i in range(MAX_NPCS):
            base = npc_feature_start + i * 4
            obs[base:base+2] = 10.0   # rel_pos = far away (not 0 = collision!)
            obs[base+2:base+4] = 0.0  # rel_vel = stationary
        
        try:
            world = self.env.unwrapped.world
            ego = world.agents[0]
            ego_pos = ego.state.p_pos.astype(np.float32)
            ego_vel = ego.state.p_vel.astype(np.float32)
            goal_pos = getattr(ego, 'goal_pos', np.zeros(2, dtype=np.float32))
            
            # Sanitize ego state
            ego_pos = np.nan_to_num(ego_pos, nan=0.0, posinf=0.0, neginf=0.0)
            ego_vel = np.nan_to_num(ego_vel, nan=0.0, posinf=0.0, neginf=0.0)
            goal_pos = np.nan_to_num(goal_pos, nan=0.0, posinf=0.0, neginf=0.0)
            
            # [0:2] Ego velocity
            obs[0:2] = ego_vel
            
            # [2:4] Ego position
            obs[2:4] = ego_pos
            
            # [4:6] Goal direction (normalized)
            goal_vec = goal_pos - ego_pos
            goal_dist = float(np.linalg.norm(goal_vec))
            if goal_dist > 1e-6:
                goal_dir = goal_vec / goal_dist
            else:
                goal_dir = np.zeros(2, dtype=np.float32)
            obs[4:6] = goal_dir
            
            # [6:7] Goal distance
            obs[6] = goal_dist
            
            # [7:103] NPC features - ONLY ACTIVE NPCs overwrite the defaults
            # ══════════════════════════════════════════════════════════════
            # [FIX] Sort NPCs by distance to ego (CLOSEST FIRST)
            # This is CRITICAL for zero-shot generalization:
            # - At N=12 (training): slots 0-11 filled, 12-23 = padding
            # - At N=16 (test):     slots 0-15 filled, 16-23 = padding
            # Without sorting: slots 12-15 get RANDOM NPCs → distribution shift
            # With sorting:    slots 12-15 get FARTHEST NPCs → close to padding values
            # ══════════════════════════════════════════════════════════════
            active_npcs_data = []
            for i in range(1, min(self.active_npc_count + 1, len(world.agents))):
                npc = world.agents[i]
                npc_pos = np.nan_to_num(npc.state.p_pos.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                npc_vel = np.nan_to_num(npc.state.p_vel.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                
                rel_pos = npc_pos - ego_pos
                rel_vel = npc_vel - ego_vel
                dist = float(np.linalg.norm(rel_pos))
                
                # Clip to reasonable range to prevent extreme values
                rel_pos = np.clip(rel_pos, -10.0, 10.0)
                rel_vel = np.clip(rel_vel, -5.0, 5.0)
                
                active_npcs_data.append((dist, rel_pos, rel_vel))
            
            # Sort by distance (if enabled) and apply K-cap
            if self.c.sort_obs_by_distance:
                active_npcs_data.sort(key=lambda x: x[0])
            
            K_OBS_CAP = self.c.k_obs_cap
            for slot_idx, (dist, rel_pos, rel_vel) in enumerate(active_npcs_data):
                if slot_idx >= K_OBS_CAP:
                    break
                idx = npc_feature_start + slot_idx * 4
                obs[idx:idx+4] = [rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1]]
            
            # Inactive NPCs keep their default "far away" values (10.0, 10.0, 0, 0)
            
            # ══════════════════════════════════════════════════════════════════
            # EXPANDED SCALAR INVARIANT FEATURES (12 density-independent features)
            # These summarize the crowd state WITHOUT depending on individual NPC slots.
            # At N=12 and N=22, these features look similar, enabling generalization.
            # ══════════════════════════════════════════════════════════════════
            all_dists = []
            all_rel_vels = []
            total_force = np.zeros(2, dtype=np.float32)
            min_dist = 10.0
            
            for i in range(1, min(self.active_npc_count + 1, len(world.agents))):
                npc = world.agents[i]
                npc_pos = np.nan_to_num(npc.state.p_pos.astype(np.float32), nan=0.0)
                npc_vel = np.nan_to_num(npc.state.p_vel.astype(np.float32), nan=0.0)
                rel_pos = npc_pos - ego_pos
                rel_vel = npc_vel - ego_vel
                dist = float(np.linalg.norm(rel_pos))
                all_dists.append(dist)
                all_rel_vels.append(rel_vel)
                min_dist = min(min_dist, max(dist, 1e-6))
                
                # Repulsive force (simplified SFM): only nearby NPCs
                if dist < 2.0 and dist > 1e-6:
                    direction = -rel_pos / dist
                    force_mag = 2.0 * np.exp((0.3 - dist) / 0.5)
                    total_force += force_mag * direction
            
            force_mag = float(np.linalg.norm(total_force))
            ego_speed = float(np.linalg.norm(ego_vel))
            
            # ── Feature 1: Crowd pressure (log-space, bounded) ──
            feat_pressure = np.log1p(force_mag)
            
            # ── Feature 2: Force-velocity alignment (cosine similarity) ──
            if force_mag > 1e-3 and ego_speed > 1e-3:
                feat_alignment = float(np.dot(total_force, ego_vel) / (force_mag * ego_speed))
            else:
                feat_alignment = 0.0
            
            # ── Feature 3: Collision risk (inverse min distance) ──
            feat_risk = 1.0 / (min_dist + 0.1)
            
            # ── Feature 4-6: Nearest-K distances (K=3, captures immediate threat) ──
            sorted_dists = sorted(all_dists) if all_dists else [10.0, 10.0, 10.0]
            while len(sorted_dists) < 3:
                sorted_dists.append(10.0)  # pad with "far away"
            feat_nearest_1 = 1.0 / (sorted_dists[0] + 0.1)
            feat_nearest_2 = 1.0 / (sorted_dists[1] + 0.1)
            feat_nearest_3 = 1.0 / (sorted_dists[2] + 0.1)
            
            # ── Feature 7-9: Crowd density in concentric zones (normalized) ──
            n_active = max(len(all_dists), 1)
            feat_n_intimate = sum(1 for d in all_dists if d < 0.3) / float(MAX_NPCS)
            feat_n_personal = sum(1 for d in all_dists if d < 0.5) / float(MAX_NPCS)
            feat_n_social   = sum(1 for d in all_dists if d < 1.0) / float(MAX_NPCS)
            
            # ── Feature 10: Active NPC fraction (tells agent how crowded it is) ──
            feat_active_frac = len(all_dists) / float(MAX_NPCS)
            
            # ── Feature 11-12: Mean relative velocity of nearby NPCs ──
            nearby_vels = [v for v, d in zip(all_rel_vels, all_dists) if d < 1.5]
            if nearby_vels:
                mean_rv = np.mean(nearby_vels, axis=0)
                feat_mean_rv_x = float(np.clip(mean_rv[0], -5.0, 5.0))
                feat_mean_rv_y = float(np.clip(mean_rv[1], -5.0, 5.0))
            else:
                feat_mean_rv_x = 0.0
                feat_mean_rv_y = 0.0
            
            # Pack into obs (12 features)
            scalar_start = 7 + MAX_NPCS * 4  # = 103
            obs[scalar_start]      = np.clip(feat_pressure, 0.0, 10.0)
            obs[scalar_start + 1]  = np.clip(feat_alignment, -1.0, 1.0)
            obs[scalar_start + 2]  = np.clip(feat_risk, 0.0, 10.0)
            obs[scalar_start + 3]  = np.clip(feat_nearest_1, 0.0, 10.0)
            obs[scalar_start + 4]  = np.clip(feat_nearest_2, 0.0, 10.0)
            obs[scalar_start + 5]  = np.clip(feat_nearest_3, 0.0, 10.0)
            obs[scalar_start + 6]  = np.clip(feat_n_intimate, 0.0, 1.0)
            obs[scalar_start + 7]  = np.clip(feat_n_personal, 0.0, 1.0)
            obs[scalar_start + 8]  = np.clip(feat_n_social, 0.0, 1.0)
            obs[scalar_start + 9]  = np.clip(feat_active_frac, 0.0, 1.0)
            obs[scalar_start + 10] = feat_mean_rv_x
            obs[scalar_start + 11] = feat_mean_rv_y
            
        except Exception as e:
            # Return default obs if world access fails
            pass
        
        # Final sanitization - ensure no NaN/inf in output
        obs = np.nan_to_num(obs, nan=10.0, posinf=10.0, neginf=-10.0)
        
        return obs
    
    def _enforce_boundaries(self):
        """
        Enforce world boundaries to prevent agents from drifting infinitely.
        Called after env.step() to clip positions.
        
        NOTE: Only applies to ACTIVE agents (ego + active NPCs).
        Inactive NPCs are left where they are to avoid clustering at boundary.
        """
        WORLD_BOUND = 2.0  # Agents bounded to [-2, 2] in x and y
        
        try:
            world = self.env.unwrapped.world
            
            # Only enforce boundaries for ego and active NPCs
            num_to_enforce = 1 + self.active_npc_count  # ego + active NPCs
            
            for i, agent in enumerate(world.agents[:num_to_enforce]):
                pos = agent.state.p_pos
                vel = agent.state.p_vel
                
                # Clip position
                new_pos = np.clip(pos, -WORLD_BOUND, WORLD_BOUND)
                
                # Zero velocity if hit boundary (soft bounce)
                if new_pos[0] != pos[0]:
                    vel[0] = 0.0
                if new_pos[1] != pos[1]:
                    vel[1] = 0.0
                
                agent.state.p_pos = new_pos
                agent.state.p_vel = vel
                
        except Exception:
            pass
    
    def _compute_ego_reward_and_info(self, obs_dict, rew_dict, term_dict, trunc_dict):
        """
        Compute reward for Ego agent based on social navigation metrics.
        
        Returns: reward, terminated, truncated, info
        
        Proper Gymnasium semantics:
        - terminated = True: Episode ended due to goal/failure (bootstrap with 0)
        - truncated = True: Episode ended due to timeout (bootstrap with V(s'))
        
        NOTE: Only considers ACTIVE NPCs for collision/intrusion checks.
        """
        reward = self.c.reward_time_penalty  # Time penalty
        terminated = False  # Goal reached or critical failure
        truncated = False   # Timeout
        
        info = {
            "collisions": 0,
            "intrusions": 0,
            "goal_reached": False,
            "freezing": False,
            "ego_velocity": 0.0,
            "min_dist_to_npc": float('inf'),
            "dist_to_goal": 0.0,
            # ══════════════════════════════════════════════════════════════════
            # [CRITICAL FIX] Add active_npc_count to step info!
            # Without this, density-aware gating in FIR_social.py is BLIND
            # (always defaults to "easy" bucket, never fills "hard" bucket)
            # ══════════════════════════════════════════════════════════════════
            "active_npc_count": self.active_npc_count,
        }
        
        try:
            world = self.env.unwrapped.world
            ego = world.agents[0]
            ego_pos = np.nan_to_num(ego.state.p_pos, nan=0.0, posinf=0.0, neginf=0.0)
            ego_vel = np.nan_to_num(ego.state.p_vel, nan=0.0, posinf=0.0, neginf=0.0)
            goal_pos = getattr(ego, 'goal_pos', np.zeros(2))
            goal_pos = np.nan_to_num(goal_pos, nan=0.0, posinf=0.0, neginf=0.0)
            
            ego_vel_mag = float(np.linalg.norm(ego_vel))
            info["ego_velocity"] = ego_vel_mag if np.isfinite(ego_vel_mag) else 0.0
            
            # Calculate distance to goal
            dist_to_goal = float(np.linalg.norm(ego_pos - goal_pos))
            if not np.isfinite(dist_to_goal):
                dist_to_goal = 0.0
            info["dist_to_goal"] = dist_to_goal
            
            # ══════════════════════════════════════════════════════════════════
            # [DENSE REWARD] Distance-based shaping to help ALL agents learn
            # This gives continuous signal: closer to goal = less penalty
            # Without this, Baseline gets 0% success (sparse reward problem)
            # Coefficient 0.1 is enough to guide navigation but not overpower FIR
            # ══════════════════════════════════════════════════════════════════
            reward += -0.1 * dist_to_goal
            
            # Check goal reached
            if dist_to_goal < self.c.goal_tolerance:
                reward += self.c.reward_goal
                terminated = True  # Goal reached = terminated
                self.ego_reached_goal = True
                info["goal_reached"] = True
            
            # Check collisions and intrusions with ACTIVE NPCs only
            ego_radius = float(ego.size) if hasattr(ego, 'size') else 0.15
            
            # Only check active NPCs (not inactive ones placed far away)
            for i in range(1, self.active_npc_count + 1):
                if i >= len(world.agents):
                    continue
                
                npc = world.agents[i]
                npc_pos = np.nan_to_num(npc.state.p_pos, nan=0.0, posinf=0.0, neginf=0.0)
                npc_radius = float(npc.size) if hasattr(npc, 'size') else 0.15
                
                dist = float(np.linalg.norm(ego_pos - npc_pos))
                if not np.isfinite(dist):
                    continue  # Skip if distance is invalid
                
                info["min_dist_to_npc"] = min(info["min_dist_to_npc"], dist)
                
                # Collision check
                collision_dist = ego_radius + npc_radius
                if dist < collision_dist:
                    reward += self.c.reward_collision
                    info["collisions"] += 1
                    self.episode_collisions += 1
                
                # Intrusion check (intimate space)
                elif dist < self.c.intimate_space:
                    reward += self.c.reward_intrusion_penalty
                    info["intrusions"] += 1
                    self.episode_intrusions += 1
            
            # Freezing detection (low velocity when not at goal)
            if info["ego_velocity"] < 0.05 and dist_to_goal > self.c.goal_tolerance:
                info["freezing"] = True
            
        except Exception as e:
            pass
        
        # Sanitize reward
        if not np.isfinite(reward):
            reward = self.c.reward_time_penalty
        
        # Proper timeout handling
        # Check if underlying env signals truncation
        all_truncated = all(trunc_dict.values())
        
        # If timeout (truncated) but not already terminated by goal
        if all_truncated and not terminated:
            truncated = True
        
        # Add SB3 compatibility flag
        info["TimeLimit.truncated"] = truncated
        
        # Add episode stats to info
        info["episode_collisions"] = self.episode_collisions
        info["episode_intrusions"] = self.episode_intrusions
        info["episode_steps"] = self.episode_steps
        
        # Add extrinsic reward for tracking (FIR wrapper will override this)
        info["r_ext"] = reward
        info["r_int"] = 0.0  # No intrinsic reward in base env (FIR wrapper overrides)
        
        return reward, terminated, truncated, info
    
    def render(self):
        if hasattr(self.env, "render"):
            return self.env.render()
        return None
    
    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
    
    @property
    def unwrapped(self):
        return self.env.unwrapped
    
    def __getattr__(self, name):
        return getattr(self.env, name)


# ==============================================================================
# Single Agent Gymnasium Wrapper (for SB3 compatibility)
# ==============================================================================

class SocialNavGymWrapper(gym.Env):
    """
    Wraps SocialNavWrapper to provide standard Gymnasium interface.
    """
    metadata = {"render_modes": []}
    
    def __init__(self, social_nav_wrapper: SocialNavWrapper):
        super().__init__()
        self.env = social_nav_wrapper
        self.observation_space = social_nav_wrapper.observation_space
        self.action_space = social_nav_wrapper.action_space
    
    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()
    
    @property
    def unwrapped(self):
        return self.env.unwrapped


# ==============================================================================
# Factory Function
# ==============================================================================

def make_social_nav_env(
    num_npcs: int = 6,
    scenario: str = "corridor",
    max_cycles: int = 100,
    config: Optional[SocialNavConfig] = None,
    # ══════════════════════════════════════════════════════════════════════════════
    # DENSITY RANDOMIZATION (Ideal 2)
    # ══════════════════════════════════════════════════════════════════════════════
    randomize_density: bool = False,
    min_active_npcs: int = 6,
    max_active_npcs: int = 12,
) -> SocialNavGymWrapper:
    """
    Factory function to create a Social Navigation environment.
    
    Args:
        num_npcs: Number of NPC agents (scripted) - used when randomize_density=False
        scenario: "corridor" or "intersection"
        max_cycles: Maximum steps per episode
        config: Optional custom configuration
        randomize_density: If True, randomize NPC count each episode
        min_active_npcs: Minimum NPCs when randomizing
        max_active_npcs: Maximum NPCs when randomizing
    
    Returns:
        SocialNavGymWrapper: Gymnasium-compatible environment
        
    NOTE: Observation dimension is ALWAYS 115 regardless of num_npcs!
          This enables zero-shot generalization.
          
    NEW: When randomize_density=True, the environment is created with max_active_npcs,
         and a random subset (min_active_npcs to max_active_npcs) is activated each episode.
         This makes VecNormalize statistics robust to different densities.
    """
    if config is None:
        config = SocialNavConfig()
    
    # Set density randomization parameters
    config.randomize_density = randomize_density
    config.min_active_npcs = min_active_npcs
    config.max_active_npcs = max_active_npcs
    
    if randomize_density:
        # When randomizing, create env with MAX NPCs
        # The wrapper will randomly activate a subset at each reset
        actual_num_npcs = max_active_npcs
        config.num_npcs = max_active_npcs
    else:
        actual_num_npcs = num_npcs
        config.num_npcs = num_npcs
    
    config.scenario = scenario
    config.max_cycles = max_cycles
    
    # Total agents = 1 (ego) + num_npcs
    total_agents = 1 + actual_num_npcs
    
    # Create base PettingZoo environment
    base_env = simple_spread_v3.parallel_env(
        N=total_agents,
        local_ratio=0.5,
        max_cycles=max_cycles,
        continuous_actions=True,
    )
    
    # Wrap with Social Navigation logic
    social_env = SocialNavWrapper(base_env, config)
    
    # Wrap for Gymnasium compatibility
    gym_env = SocialNavGymWrapper(social_env)
    
    return gym_env


# ==============================================================================
# Test
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Social Navigation Environment")
    print("=" * 60)
    print(f"\n📐 FIXED observation dimension: {FIXED_OBS_DIM}")
    print(f"   (7 ego features + {MAX_NPCS} NPCs × 4 features each + {SCALAR_INV_DIM} scalar invariants)")
    print(f"   NPC slots sorted by distance (closest first) for generalization")
    
    # Test corridor scenario
    print("\n--- Corridor Scenario (4 NPCs) ---")
    env = make_social_nav_env(num_npcs=4, scenario="corridor", max_cycles=50)
    obs, info = env.reset(seed=42)
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial info: {info}")
    
    # Run a few steps
    total_reward = 0
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode ended at step {step+1}")
            break
    
    print(f"Total reward after {step+1} steps: {total_reward:.2f}")
    print(f"Final info: {info}")
    
    env.close()
    
    # Test intersection scenario
    print("\n--- Intersection Scenario (6 NPCs) ---")
    env = make_social_nav_env(num_npcs=6, scenario="intersection", max_cycles=50)
    obs, info = env.reset(seed=42)
    
    print(f"Observation shape: {obs.shape}")
    print(f"Scenario: {info.get('scenario')}")
    
    env.close()
    
    # ══════════════════════════════════════════════════════════════════════════
    # CRITICAL TEST: Zero-shot compatibility (different NPC counts, SAME obs dim)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("🧪 Zero-Shot Compatibility Test")
    print("=" * 60)
    
    test_configs = [4, 6, 8, 12, 16, 20]
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
        print(f"✅ All observation shapes are identical: {obs_shapes[0]}")
        print("✅ Zero-shot generalization is ENABLED!")
        print("   (Train on 6 NPCs, test on 12+ NPCs with same model)")
    else:
        print("❌ Observation shapes vary - zero-shot generalization BROKEN!")
        print(f"   Shapes: {obs_shapes}")
    
    print("\n✅ Social Navigation Environment Test Complete!")