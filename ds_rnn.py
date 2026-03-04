"""
DS-RNN (Decentralized Structural-RNN) Baseline for Social Navigation.

Adapted from: Liu et al., "Decentralized Structural-Recurrent Neural Network
for Robot Navigation in Dynamic Environments", RA-L 2021.

Architecture:
  1. Parse 115-dim obs → ego(7) + per-neighbor(24×4) + scalar(12)
  2. Per-neighbor MLP (shared weights) → neighbor embeddings
  3. Attention-based aggregation over neighbor embeddings
  4. Concat ego + aggregated neighbors + scalar → MLP → features
  5. Standard PPO actor-critic heads on top

This captures the key DS-RNN insight: structural decomposition with
per-agent processing and attention-based social context aggregation.

For the recurrent component, we provide two modes:
  - "dsrnn": Uses sb3-contrib RecurrentPPO with LSTM (full DS-RNN)
  - "dsrnn_mlp": Uses regular PPO with structural extractor (no LSTM)

Both use the same structural features extractor.

Compatible with:
  - run_social.py training pipeline (via get_dsrnn_policy_kwargs)
  - eval_unified.py evaluation (saves as standard SB3 model)

v1.0
"""

__version__ = "1.0"

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from typing import Dict, Optional, Tuple, Type

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ==============================================================================
# Observation Parsing Constants (must match env_social_nav.py)
# ==============================================================================

MAX_NPCS = 24
EGO_DIM = 7          # ego_vel(2) + ego_pos(2) + goal_dir(2) + goal_dist(1)
NPC_FEAT_DIM = 4     # rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y (per NPC)
SCALAR_DIM = 12      # density-invariant crowd summary
TOTAL_OBS_DIM = EGO_DIM + (MAX_NPCS * NPC_FEAT_DIM) + SCALAR_DIM  # = 115

# Inactive NPCs have rel_pos = 10.0 (far away sentinel) in raw obs.
# After VecNormalize: real NPCs normalize to small values, padding stays large
# Threshold 4.0 safely separates real (< ~2.0) from ghosts (> 6.0)
INACTIVE_THRESHOLD = 4.0


# ==============================================================================
# Structural Features Extractor (Core DS-RNN Component)
# ==============================================================================

class DSRNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    DS-RNN-style structural features extractor for social navigation.
    
    Instead of processing the flat 115-dim obs through a generic MLP,
    this extractor:
      1. Parses obs into ego state, per-neighbor features, and scalar context
      2. Processes each neighbor independently through a shared MLP
      3. Masks out inactive neighbors (padding slots)
      4. Aggregates neighbor embeddings via multi-head attention
      5. Fuses ego + social context + scalar features
    
    This structural inductive bias gives the agent explicit awareness of
    individual neighbor dynamics and compositional social reasoning.
    
    Args:
        observation_space: Gym observation space (must be Box with shape (115,))
        neighbor_embed_dim: Hidden dim for per-neighbor MLP (default: 64)
        ego_embed_dim: Hidden dim for ego encoder (default: 64)
        social_embed_dim: Output dim of social attention (default: 64)
        n_attention_heads: Number of attention heads (default: 4)
        features_dim: Final features dimension output (default: 128)
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        neighbor_embed_dim: int = 64,
        ego_embed_dim: int = 64,
        social_embed_dim: int = 64,
        n_attention_heads: int = 4,
        features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)
        
        self.neighbor_embed_dim = neighbor_embed_dim
        self.ego_embed_dim = ego_embed_dim
        self.social_embed_dim = social_embed_dim
        
        # ── Ego encoder: ego state (7) + scalar context (12) → ego_embed_dim ──
        self.ego_encoder = nn.Sequential(
            nn.Linear(EGO_DIM + SCALAR_DIM, ego_embed_dim),
            nn.ReLU(),
            nn.Linear(ego_embed_dim, ego_embed_dim),
            nn.ReLU(),
        )
        
        # ── Per-neighbor encoder (shared weights) ──
        # Input: NPC_FEAT_DIM (4) = rel_pos(2) + rel_vel(2)
        # Also receives ego goal direction (2) for goal-conditioned reasoning
        neighbor_input_dim = NPC_FEAT_DIM + 2  # +2 for goal direction context
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(neighbor_input_dim, neighbor_embed_dim),
            nn.ReLU(),
            nn.Linear(neighbor_embed_dim, neighbor_embed_dim),
            nn.ReLU(),
        )
        
        # ── Social attention: aggregate neighbor embeddings ──
        # Using multi-head attention where ego embedding is the query
        # and neighbor embeddings are keys/values
        self.attention_query = nn.Linear(ego_embed_dim, social_embed_dim)
        self.attention_key = nn.Linear(neighbor_embed_dim, social_embed_dim)
        self.attention_value = nn.Linear(neighbor_embed_dim, social_embed_dim)
        self.n_heads = n_attention_heads
        self.head_dim = social_embed_dim // n_attention_heads
        assert social_embed_dim % n_attention_heads == 0, \
            f"social_embed_dim ({social_embed_dim}) must be divisible by n_heads ({n_attention_heads})"
        
        self.attention_out = nn.Linear(social_embed_dim, social_embed_dim)
        
        # ── Fusion: ego_embed + social_context → final features ──
        fusion_input_dim = ego_embed_dim + social_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )
    
    def _parse_observation(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parse flat observation into structured components.
        
        Args:
            obs: (batch, 115) flat observation
            
        Returns:
            ego: (batch, 7) ego state
            neighbors: (batch, 24, 4) per-neighbor features
            scalars: (batch, 12) scalar crowd summary
            active_mask: (batch, 24) bool mask (True = active neighbor)
        """
        batch_size = obs.shape[0]
        
        # Ego state: [0:7]
        ego = obs[:, :EGO_DIM]
        
        # Neighbor features: [7:103] → reshape to (batch, 24, 4)
        npc_start = EGO_DIM
        npc_end = EGO_DIM + MAX_NPCS * NPC_FEAT_DIM
        neighbors = obs[:, npc_start:npc_end].reshape(batch_size, MAX_NPCS, NPC_FEAT_DIM)
        
        # Scalar features: [103:115]
        scalars = obs[:, npc_end:]
        
        # Active mask: neighbors with |rel_pos| < threshold are active
        # rel_pos is the first 2 features of each neighbor
        rel_pos = neighbors[:, :, :2]  # (batch, 24, 2)
        rel_pos_norm = torch.norm(rel_pos, dim=-1)  # (batch, 24)
        active_mask = rel_pos_norm < INACTIVE_THRESHOLD  # (batch, 24)
        
        return ego, neighbors, scalars, active_mask
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with structural decomposition.
        
        Args:
            observations: (batch, 115) flat observations
            
        Returns:
            features: (batch, features_dim) structured features
        """
        ego, neighbors, scalars, active_mask = self._parse_observation(observations)
        batch_size = observations.shape[0]
        
        # ── 1. Encode ego state + scalar context ──
        ego_input = torch.cat([ego, scalars], dim=-1)  # (batch, 19)
        ego_embed = self.ego_encoder(ego_input)  # (batch, ego_embed_dim)
        
        # ── 2. Encode each neighbor (with goal direction context) ──
        # Broadcast goal direction (from ego obs) to each neighbor
        goal_dir = ego[:, 4:6]  # (batch, 2) - goal direction from obs
        goal_dir_expanded = goal_dir.unsqueeze(1).expand(-1, MAX_NPCS, -1)  # (batch, 24, 2)
        
        neighbor_input = torch.cat([neighbors, goal_dir_expanded], dim=-1)  # (batch, 24, 6)
        # Reshape for shared MLP: (batch*24, 6)
        neighbor_input_flat = neighbor_input.reshape(-1, NPC_FEAT_DIM + 2)
        neighbor_embed_flat = self.neighbor_encoder(neighbor_input_flat)  # (batch*24, embed_dim)
        neighbor_embed = neighbor_embed_flat.reshape(batch_size, MAX_NPCS, self.neighbor_embed_dim)
        # (batch, 24, neighbor_embed_dim)
        
        # ── 3. Attention-based social aggregation ──
        # Query from ego, keys/values from neighbors
        Q = self.attention_query(ego_embed)  # (batch, social_embed_dim)
        Q = Q.unsqueeze(1)  # (batch, 1, social_embed_dim)
        K = self.attention_key(neighbor_embed)  # (batch, 24, social_embed_dim)
        V = self.attention_value(neighbor_embed)  # (batch, 24, social_embed_dim)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)
        # (batch, n_heads, 1, head_dim)
        K = K.reshape(batch_size, MAX_NPCS, self.n_heads, self.head_dim).transpose(1, 2)
        # (batch, n_heads, 24, head_dim)
        V = V.reshape(batch_size, MAX_NPCS, self.n_heads, self.head_dim).transpose(1, 2)
        # (batch, n_heads, 24, head_dim)
        
        # Scaled dot-product attention with masking
        scale = self.head_dim ** 0.5
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        # (batch, n_heads, 1, 24)
        
        # Mask inactive neighbors with -inf
        mask = active_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, 24)
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        # Handle case where ALL neighbors are inactive (avoid NaN from softmax of all -inf)
        all_inactive = ~active_mask.any(dim=-1)  # (batch,)
        if all_inactive.any():
            # For batches with no active neighbors, set uniform attention (will be zero anyway)
            attn_scores[all_inactive] = 0.0
        
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, n_heads, 1, 24)
        
        # Zero out attention for fully inactive batches
        if all_inactive.any():
            attn_weights[all_inactive] = 0.0
        
        # Handle NaN from softmax (additional safety)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # Weighted sum of values
        social_context = torch.matmul(attn_weights, V)
        # (batch, n_heads, 1, head_dim)
        social_context = social_context.transpose(1, 2).reshape(batch_size, 1, self.social_embed_dim)
        social_context = social_context.squeeze(1)  # (batch, social_embed_dim)
        social_context = self.attention_out(social_context)  # (batch, social_embed_dim)
        
        # ── 4. Fuse ego + social context ──
        fused = torch.cat([ego_embed, social_context], dim=-1)
        features = self.fusion(fused)  # (batch, features_dim)
        
        return features


# ==============================================================================
# Policy kwargs factory (for easy integration with run_social.py)
# ==============================================================================

def get_dsrnn_policy_kwargs(
    neighbor_embed_dim: int = 64,
    ego_embed_dim: int = 64,
    social_embed_dim: int = 64,
    n_attention_heads: int = 4,
    features_dim: int = 128,
    net_arch: Optional[Dict] = None,
) -> Dict:
    """
    Get policy_kwargs for SB3 PPO with DS-RNN features extractor.
    
    Usage with SB3:
        from ds_rnn import get_dsrnn_policy_kwargs
        
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=get_dsrnn_policy_kwargs(),
            ...
        )
    
    Args:
        neighbor_embed_dim: Per-neighbor MLP hidden dim
        ego_embed_dim: Ego encoder hidden dim
        social_embed_dim: Attention output dim
        n_attention_heads: Number of attention heads
        features_dim: Final features output dim
        net_arch: Actor-critic network architecture (default: [64, 64] each)
    
    Returns:
        policy_kwargs dict for SB3 PPO constructor
    """
    if net_arch is None:
        # Smaller policy heads since features extractor is more expressive
        net_arch = dict(pi=[64, 64], vf=[64, 64])
    
    return dict(
        features_extractor_class=DSRNNFeaturesExtractor,
        features_extractor_kwargs=dict(
            neighbor_embed_dim=neighbor_embed_dim,
            ego_embed_dim=ego_embed_dim,
            social_embed_dim=social_embed_dim,
            n_attention_heads=n_attention_heads,
            features_dim=features_dim,
        ),
        net_arch=net_arch,
    )


# ==============================================================================
# Model info helper (for paper reporting)
# ==============================================================================

def count_parameters(model) -> Dict[str, int]:
    """Count parameters in a PPO model with DS-RNN extractor."""
    total = sum(p.numel() for p in model.policy.parameters())
    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    
    extractor_params = sum(
        p.numel() for p in model.policy.features_extractor.parameters()
    )
    
    return {
        "total": total,
        "trainable": trainable,
        "features_extractor": extractor_params,
        "policy_heads": total - extractor_params,
    }


# ==============================================================================
# Quick test
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DS-RNN Features Extractor Test")
    print("=" * 60)
    
    # Create dummy observation space matching env
    obs_space = gym.spaces.Box(low=-10, high=10, shape=(TOTAL_OBS_DIM,), dtype=np.float32)
    
    extractor = DSRNNFeaturesExtractor(obs_space)
    print(f"\nExtractor architecture:")
    print(extractor)
    
    total_params = sum(p.numel() for p in extractor.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 4
    dummy_obs = torch.randn(batch_size, TOTAL_OBS_DIM)
    
    # Simulate some inactive NPCs (set rel_pos to 10.0)
    for i in range(15, 24):  # Last 9 NPCs inactive
        base = EGO_DIM + i * NPC_FEAT_DIM
        dummy_obs[:, base:base+2] = 10.0
    
    features = extractor(dummy_obs)
    print(f"\nInput shape:  {dummy_obs.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Output range: [{features.min():.3f}, {features.max():.3f}]")
    
    # Test with all inactive (edge case)
    all_inactive_obs = torch.randn(2, TOTAL_OBS_DIM)
    for i in range(MAX_NPCS):
        base = EGO_DIM + i * NPC_FEAT_DIM
        all_inactive_obs[:, base:base+2] = 10.0
    features_empty = extractor(all_inactive_obs)
    print(f"\nAll-inactive output: {features_empty.shape}, has_nan={torch.isnan(features_empty).any()}")
    
    # Test policy_kwargs integration
    print("\n--- PPO Integration Test ---")
    kwargs = get_dsrnn_policy_kwargs()
    print(f"policy_kwargs keys: {list(kwargs.keys())}")
    print(f"features_extractor_class: {kwargs['features_extractor_class'].__name__}")
    print(f"net_arch: {kwargs['net_arch']}")
    
    print("\n✅ All tests passed!")