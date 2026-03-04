# PSS-Social: Potential-Based Social Shaping for Safe Navigation in Dense Crowds

Code for our  2026 submission on density-generalizable social robot navigation using deep reinforcement learning.

## Overview

Existing DRL methods for social navigation fail when tested at crowd densities beyond their training distribution. We identify two key factors behind this failure: observation distribution shift from learned normalization statistics, and reward designs that either under-penalize collisions or induce conservative freezing.

PSS-Social addresses these through:
- **Density-invariant observation encoding**: KNN-sorted, fixed-length neighbor slots with bounded crowd summaries that maintain stable input distributions as crowd size increases.
- **Potential-based social shaping**: Proxemic zone penalties formulated as a potential function, providing smooth anticipatory gradients for socially compliant navigation without altering the optimal policy.

Our method achieves 86.4% collision-free success at 2.33 ped/m², a density 31% beyond the training maximum, where state-of-the-art attention-based methods retain only a fraction of their in-distribution performance.

## Project Structure

| File | Description |
|------|-------------|
| `pss_social.py` | PSS reward shaping module and experiment configs |
| `env_social_nav.py` | Social navigation environment with density randomization |
| `run_social.py` | Training script for PSS-Social variants |
| `train_baselines.py` | Training for SARL and LSTM-RL baselines |
| `eval_unified.py` | Unified evaluation across density sweep |
| `eval_baselines.py` | Evaluation for SARL and LSTM-RL |
| `ds_rnn.py` | DS-RNN baseline architecture |
| `policies_analytic.py` | ORCA and SFM analytic baselines |
| `plot_.py` | Figure generation for the paper |
| `visualize_social.py` | Episode visualization and video rendering |

## Requirements

- Python 3.8+
- PyTorch
- Stable-Baselines3
- sb3-contrib (for LSTM-RL baseline)
- PettingZoo MPE
- Numba (optional, for SFM acceleration)

## Training

```bash
# Train PSS-Social
python run_social.py --experiment PSS_Social --seeds 9,10,42,123,456 --timesteps 1000000

# Train baselines
python train_baselines.py --agent SARL,LSTM_RL --seeds 9,10,42,123,456 --timesteps 10000000
```

## Evaluation

```bash
# Evaluate across density sweep
python eval_unified.py ./runs_social --scenario random --densities 11,13,15,17,19,21 -n 100
```