# Experiment Directory Structure

## Overview

Each experiment run creates a timestamped directory with all results organized by experiment type.

## Directory Structure

```
delay_experiments/
└── experiments_YYYYMMDD_HHMMSS/
    ├── checkpoints/
    │   ├── realtime_baseline/
    │   │   ├── sac_policy_ep100.pth
    │   │   ├── sac_q1_ep100.pth
    │   │   ├── sac_q2_ep100.pth
    │   │   ├── sac_policy_ep200.pth
    │   │   ├── sac_q1_ep200.pth
    │   │   ├── sac_q2_ep200.pth
    │   │   ├── ... (every 100 episodes)
    │   │   ├── sac_policy_final.pth
    │   │   ├── sac_q1_final.pth
    │   │   └── sac_q2_final.pth
    │   ├── gym_mode_0.0s/
    │   │   ├── sac_policy_ep100.pth
    │   │   ├── ... (same structure)
    │   │   └── sac_policy_final.pth
    │   ├── gym_mode_0.05s/
    │   │   └── ...
    │   ├── gym_mode_0.1s/
    │   │   └── ...
    │   ├── gym_mode_0.15s/
    │   │   └── ...
    │   ├── gym_mode_0.2s/
    │   │   └── ...
    │   ├── gym_mode_0.25s/
    │   │   └── ...
    │   ├── gym_mode_0.3s/
    │   │   └── ...
    │   └── gym_mode_1.0s/
    │       └── ...
    ├── metrics/
    │   ├── realtime_baseline/
    │   │   ├── metrics/
    │   │   │   └── training_metrics.json
    │   │   └── plots/
    │   │       ├── episode_rewards.png
    │   │       ├── episode_lengths.png
    │   │       ├── losses.png
    │   │       └── ...
    │   ├── gym_mode_0.0s/
    │   │   └── ... (same structure)
    │   ├── gym_mode_0.05s/
    │   │   └── ...
    │   └── ... (one directory per experiment)
    └── results/
        ├── comparison_results.json
        └── comparison_plots/
            ├── reward_comparison.png
            ├── length_comparison.png
            └── ...
```

## Naming Convention

### Checkpoints
- **Periodic checkpoints**: `sac_policy_ep{episode_num}.pth` (e.g., `sac_policy_ep100.pth`)
- **Final checkpoint**: `sac_policy_final.pth`
- Same pattern for Q-networks: `sac_q1_ep{episode_num}.pth`, `sac_q2_ep{episode_num}.pth`

### Metrics
- **JSON file**: `training_metrics.json` (in each experiment's metrics directory)
- **Plots**: Various PNG files in the `plots/` subdirectory

### Experiment Names
- **Real-time**: `realtime_baseline`
- **Gym mode**: `gym_mode_{delay}s` (e.g., `gym_mode_0.05s`, `gym_mode_1.0s`)

## Key Points

✅ **No Overwriting**: Each experiment has its own subdirectory
✅ **Clear Naming**: Experiment names include delay values
✅ **Organized Structure**: Checkpoints, metrics, and results are separated
✅ **Timestamped**: Each run gets a unique timestamp directory
✅ **Complete History**: All checkpoints (every 100 episodes) are saved

## Example Paths

For delay 0.1s experiment:
- Checkpoints: `delay_experiments/experiments_20240101_120000/checkpoints/gym_mode_0.1s/`
- Metrics: `delay_experiments/experiments_20240101_120000/metrics/gym_mode_0.1s/`
- Final checkpoint: `delay_experiments/experiments_20240101_120000/checkpoints/gym_mode_0.1s/sac_policy_final.pth`

## Loading a Checkpoint

To load a specific checkpoint:
```bash
python src/sac_agent.py \
    --policy_checkpoint delay_experiments/experiments_20240101_120000/checkpoints/gym_mode_0.1s/sac_policy_ep300.pth \
    --q1_checkpoint delay_experiments/experiments_20240101_120000/checkpoints/gym_mode_0.1s/sac_q1_ep300.pth \
    --q2_checkpoint delay_experiments/experiments_20240101_120000/checkpoints/gym_mode_0.1s/sac_q2_ep300.pth \
    --start_episode 300
```


