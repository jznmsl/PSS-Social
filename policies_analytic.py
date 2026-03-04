"""
Analytic Baseline Policies for Social Navigation Evaluation.

Provides SFM and ORCA agents that use the SAME env interface as PPO,
enabling fair comparison with identical metrics.

CRITICAL: These agents use "oracle" world state (privileged info).
          This must be noted in the paper!

Action format: 5-dim [-1, 1] (wrapper converts to MPE [0, 1] internally)
  - [no_action, left, right, down, up]

v1.0 - Initial implementation
"""

__version__ = "1.0"

import numpy as np
from typing import Optional, Tuple

from env_social_nav import SFM_NPC_Controller, SocialNavConfig


# ==============================================================================
# Base Class
# ==============================================================================

class AnalyticPolicy:
    """
    Base class for analytic navigation policies.
    
    Mimics SB3's model.predict() interface so eval code works unchanged.
    Uses privileged world state (not observations) for action computation.
    """
    
    def __init__(self, vec_env):
        """
        Args:
            vec_env: DummyVecEnv wrapping SocialNavGymWrapper
                     (NOT VecNormalize — analytic agents don't use obs)
        """
        self.vec_env = vec_env
    
    def _get_world_state(self):
        """Extract world state from the underlying PettingZoo env."""
        # Navigate through: DummyVecEnv -> SocialNavGymWrapper -> SocialNavWrapper -> MPE
        try:
            inner_env = self.vec_env.envs[0]  # DummyVecEnv -> GymWrapper
            world = inner_env.unwrapped.world
            
            ego = world.agents[0]
            ego_pos = np.nan_to_num(ego.state.p_pos, nan=0.0).copy()
            ego_vel = np.nan_to_num(ego.state.p_vel, nan=0.0).copy()
            goal_pos = np.nan_to_num(getattr(ego, 'goal_pos', np.zeros(2)), nan=0.0).copy()
            
            others_pos = []
            others_vel = []
            others_radius = []
            
            for agent in world.agents[1:]:
                pos = np.nan_to_num(agent.state.p_pos, nan=0.0)
                # Skip inactive NPCs (placed at 1000, 1000)
                if np.linalg.norm(pos) > 500:
                    continue
                others_pos.append(pos.copy())
                others_vel.append(np.nan_to_num(agent.state.p_vel, nan=0.0).copy())
                others_radius.append(getattr(agent, 'size', 0.15))
            
            return ego_pos, ego_vel, goal_pos, others_pos, others_vel, others_radius
        except Exception as e:
            # Fallback: return zeros (agent will do nothing)
            return np.zeros(2), np.zeros(2), np.zeros(2), [], [], []
    
    def _force_to_mpe_action(self, force_x: float, force_y: float) -> np.ndarray:
        """
        Convert 2D force/velocity to 5-dim MPE action in [-1, 1] range.
        
        MPE continuous action: [no_action, left, right, down, up]
        Each in [0, 1] natively, but our wrapper expects [-1, 1].
        
        Conversion: wrapper_input = native_action * 2 - 1
        """
        # Convert force to native MPE [0, 1] format
        native = np.zeros(5, dtype=np.float32)
        native[1] = max(0.0, -force_x)   # Left (negative x)
        native[2] = max(0.0, force_x)    # Right (positive x)
        native[3] = max(0.0, -force_y)   # Down (negative y)
        native[4] = max(0.0, force_y)    # Up (positive y)
        
        # Clip to [0, 1]
        native = np.clip(native, 0.0, 1.0)
        
        # Convert to wrapper's [-1, 1] range
        wrapper_action = native * 2.0 - 1.0
        
        return wrapper_action
    
    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        """
        SB3-compatible predict interface.
        
        Returns:
            action: np.array shape (1, action_dim) for VecEnv
            state: None (no RNN state)
        """
        raise NotImplementedError


# ==============================================================================
# SFM Agent (uses your existing SFM_NPC_Controller)
# ==============================================================================

class SFMPolicy(AnalyticPolicy):
    """
    Social Force Model agent for Ego.
    
    Reuses the same SFM_NPC_Controller that drives NPCs,
    so the ego agent navigates with identical physics.
    """
    
    def __init__(self, vec_env, config: Optional[SocialNavConfig] = None):
        super().__init__(vec_env)
        if config is None:
            config = SocialNavConfig()
        self.controller = SFM_NPC_Controller(config)
        self.config = config
    
    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        ego_pos, ego_vel, goal_pos, others_pos, others_vel, others_radius = \
            self._get_world_state()
        
        # Build arrays for vectorized SFM computation
        n_others = len(others_pos)
        N = 1 + n_others  # ego + others
        
        if N < 2:
            # No neighbors: just go to goal
            direction = goal_pos - ego_pos
            dist = np.linalg.norm(direction)
            if dist > 1e-5:
                force = direction / dist * self.config.npc_speed
            else:
                force = np.zeros(2)
            action = self._force_to_mpe_action(force[0], force[1])
            return np.array([action]), None
        
        all_pos = np.vstack([ego_pos.reshape(1, 2)] + [p.reshape(1, 2) for p in others_pos])
        all_vel = np.vstack([ego_vel.reshape(1, 2)] + [v.reshape(1, 2) for v in others_vel])
        all_goals = np.zeros_like(all_pos)
        all_goals[0] = goal_pos
        # Others: use current position as goal (they're managed by env's SFM)
        for i in range(n_others):
            all_goals[i + 1] = all_pos[i + 1]
        
        # Compute SFM actions for all agents, extract ego's (index 0)
        all_actions = self.controller.compute_all_actions(all_pos, all_vel, all_goals)
        sfm_action = all_actions[0]  # (5,) in [0, 1] range
        
        # Convert native [0,1] to wrapper [-1,1]
        wrapper_action = sfm_action * 2.0 - 1.0
        
        return np.array([wrapper_action]), None


# ==============================================================================
# ORCA Agent (requires Python-RVO2)
# ==============================================================================

try:
    import rvo2
    RVO2_AVAILABLE = True
except ImportError:
    RVO2_AVAILABLE = False


class ORCAPolicy(AnalyticPolicy):
    """
    Optimal Reciprocal Collision Avoidance (ORCA) agent.
    
    Requires: pip install Python-RVO2
    
    NOTE: Creates a fresh RVO simulator each step (stateless).
    This is slightly inefficient but ensures correctness.
    """
    
    def __init__(self, vec_env, config: Optional[SocialNavConfig] = None):
        super().__init__(vec_env)
        if not RVO2_AVAILABLE:
            raise ImportError(
                "ORCA requires Python-RVO2. Install with: pip install Python-RVO2\n"
                "If that fails, try: pip install pyrvo2"
            )
        if config is None:
            config = SocialNavConfig()
        self.config = config
        
        # ORCA parameters
        self.time_step = 0.1        # Should match env dt
        self.neighbor_dist = 5.0    # How far to look for neighbors
        self.max_neighbors = 20     # Max neighbors to consider
        self.time_horizon = 3.0     # Planning horizon for agents
        self.time_horizon_obst = 3.0
        self.radius = 0.15          # Agent radius
        self.max_speed = 1.0        # Max velocity
    
    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        ego_pos, ego_vel, goal_pos, others_pos, others_vel, others_radius = \
            self._get_world_state()
        
        if not others_pos:
            # No neighbors: go straight to goal
            direction = goal_pos - ego_pos
            dist = np.linalg.norm(direction)
            if dist > 1e-5:
                vel = direction / dist * self.max_speed
            else:
                vel = np.zeros(2)
            action = self._force_to_mpe_action(vel[0], vel[1])
            return np.array([action]), None
        
        # Create fresh RVO simulator
        sim = rvo2.PyRVOSimulator(
            self.time_step,
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horizon,
            self.time_horizon_obst,
            self.radius,
            self.max_speed
        )
        
        # Add ego agent (index 0)
        sim.addAgent(
            tuple(ego_pos.tolist()),
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horizon,
            self.time_horizon_obst,
            self.radius,
            self.max_speed,
            tuple(ego_vel.tolist())
        )
        
        # Add neighbor agents
        for pos, vel, rad in zip(others_pos, others_vel, others_radius):
            sim.addAgent(
                tuple(pos.tolist()),
                self.neighbor_dist,
                self.max_neighbors,
                self.time_horizon,
                self.time_horizon_obst,
                float(rad),
                self.max_speed,
                tuple(vel.tolist())
            )
        
        # Set preferred velocity (toward goal)
        disp = goal_pos - ego_pos
        dist = np.linalg.norm(disp)
        if dist > 1e-5:
            pref_vel = (disp / dist) * self.max_speed
        else:
            pref_vel = np.zeros(2)
        
        sim.setAgentPrefVelocity(0, tuple(pref_vel.tolist()))
        
        # Set other agents' preferred velocities (forward motion)
        for i in range(len(others_pos)):
            # Others just keep their current velocity direction
            v = others_vel[i]
            vmag = np.linalg.norm(v)
            if vmag > 1e-5:
                pv = (v / vmag) * min(vmag, self.max_speed)
            else:
                pv = np.zeros(2)
            sim.setAgentPrefVelocity(i + 1, tuple(pv.tolist()))
        
        # Compute ORCA velocity
        sim.doStep()
        new_vel = np.array(sim.getAgentVelocity(0))
        
        # ══════════════════════════════════════════════════════════════════
        # CRITICAL FIX: ORCA outputs a DESIRED VELOCITY, but MPE treats
        # actions as FORCE inputs (action × sensitivity × dt → acceleration).
        # Sending velocity directly causes massive overshoot + oscillation.
        #
        # Fix: Use proportional controller to convert desired velocity
        # to force command: force = gain × (v_desired - v_current)
        # gain ~2.0 works well with MPE's sensitivity=5.0, dt=0.1
        # ══════════════════════════════════════════════════════════════════
        vel_error = new_vel - ego_vel
        gain = 2.0  # Tuned for MPE physics (sensitivity=5, dt=0.1)
        force = vel_error * gain
        
        # Scale to [0, 1] action range
        force_mag = np.linalg.norm(force)
        if force_mag > 1.0:
            force = force / force_mag  # Normalize to unit magnitude
        
        action = self._force_to_mpe_action(force[0], force[1])
        
        return np.array([action]), None


# ==============================================================================
# Factory
# ==============================================================================

def get_analytic_policy(name: str, vec_env, config=None):
    """
    Factory function to get analytic policy by name.
    
    Args:
        name: "SFM" or "ORCA"
        vec_env: DummyVecEnv (no VecNormalize!)
        config: Optional SocialNavConfig
    
    Returns:
        AnalyticPolicy instance
    """
    name = name.upper()
    if name == "SFM":
        return SFMPolicy(vec_env, config)
    elif name == "ORCA":
        if not RVO2_AVAILABLE:
            raise ImportError("ORCA requires Python-RVO2. Install: pip install Python-RVO2")
        return ORCAPolicy(vec_env, config)
    else:
        raise ValueError(f"Unknown analytic policy: {name}. Choose 'SFM' or 'ORCA'")


# ==============================================================================
# Test
# ==============================================================================

if __name__ == "__main__":
    from stable_baselines3.common.vec_env import DummyVecEnv
    from env_social_nav import make_social_nav_env
    
    print("=" * 60)
    print("Testing Analytic Policies")
    print("=" * 60)
    
    # Test SFM
    print("\n--- SFM Agent ---")
    env = DummyVecEnv([lambda: make_social_nav_env(num_npcs=6, scenario="corridor")])
    sfm = SFMPolicy(env)
    
    obs = env.reset()
    for step in range(10):
        action, _ = sfm.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(f"  Step {step}: action={action[0][:3]}... reward={reward[0]:.3f}")
        if done[0]:
            obs = env.reset()
    env.close()
    print("  ✅ SFM works!")
    
    # Test ORCA (if available)
    if RVO2_AVAILABLE:
        print("\n--- ORCA Agent ---")
        env = DummyVecEnv([lambda: make_social_nav_env(num_npcs=6, scenario="corridor")])
        orca = ORCAPolicy(env)
        
        obs = env.reset()
        for step in range(10):
            action, _ = orca.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            print(f"  Step {step}: action={action[0][:3]}... reward={reward[0]:.3f}")
            if done[0]:
                obs = env.reset()
        env.close()
        print("  ✅ ORCA works!")
    else:
        print("\n⚠️ ORCA not available (pip install Python-RVO2)")
    
    print("\nDone!")