# Experiment Results Summary

## Directory Structure

Your experiment results are saved in:
```
delay_experiments/experiments_20251218_130600/
```

This directory contains three main folders:
1. **checkpoints/** - Model checkpoints
2. **metrics/** - Training metrics and plots for each experiment
3. **results/** - Comparison results across all experiments

---

## 1. Checkpoints (`checkpoints/`)

### Location
Each experiment has its own subdirectory:
- `checkpoints/gym_mode_0.01s/`
- `checkpoints/gym_mode_0.033s/`
- `checkpoints/gym_mode_0.05s/`
- `checkpoints/gym_mode_0.1s/`
- `checkpoints/gym_mode_0.5s/`
- `checkpoints/gym_mode_1.0s/`

### Contents
For each completed experiment, you'll find:
- **Periodic checkpoints**: `sac_policy_ep{N}.pth`, `sac_q1_ep{N}.pth`, `sac_q2_ep{N}.pth`
  - Saved every 100 episodes (if `save_freq=100`)
  - Example: `sac_policy_ep100.pth`, `sac_policy_ep200.pth`
  
- **Final checkpoint**: `sac_policy_final.pth`, `sac_q1_final.pth`, `sac_q2_final.pth`
  - Saved at the end of training
  - This is the model to use for evaluation/inference

- **Evaluation metrics**: `evaluation_metrics.json`
  - Contains evaluation results (avg reward, avg length, etc.)
  - Generated when the experiment was evaluated

---

## 2. Training Metrics (`metrics/`)

### Location
Each experiment has its own subdirectory:
- `metrics/gym_mode_0.01s/`
- `metrics/gym_mode_0.033s/`
- `metrics/gym_mode_0.05s/` (may be empty if training didn't complete)
- `metrics/gym_mode_0.1s/`
- `metrics/gym_mode_0.5s/`
- `metrics/gym_mode_1.0s/`

### Structure
Each metrics directory contains:
```
metrics/gym_mode_{delay}s/
├── metrics/
│   └── training_metrics.json    # All training metrics in JSON format
└── plots/
    ├── rewards.png               # Episode rewards plot
    ├── lengths.png               # Episode lengths plot
    ├── losses.png                # Training losses (Q1, Q2, Policy, Alpha)
    ├── alpha.png                 # Alpha (temperature) value over time
    ├── buffer_size.png           # Replay buffer size over time
    └── training_time.png         # Training time metrics
```

### What's in `training_metrics.json`?

The JSON file contains:
- **config**: Training configuration (hyperparameters, episode count, etc.)
- **training_time**: Total time, per-episode times, per-step times
- **episode_metrics**: 
  - `rewards`: List of episode rewards
  - `lengths`: List of episode lengths
  - `losses`: Dictionary with Q1, Q2, Policy, and Alpha losses per episode
- **step_metrics**: 
  - `rewards`: Rewards per step
  - `q1_losses`, `q2_losses`, `policy_losses`, `alpha_losses`: Losses per step
  - `alpha_values`: Alpha value per step
  - `buffer_sizes`: Replay buffer size per step
- **statistics**: Mean, std, min, max for all metrics

### Plot Descriptions

1. **rewards.png**
   - Top: Raw episode rewards + moving average (window size 100)
   - Bottom: Reward distribution histogram

2. **lengths.png**
   - Top: Episode lengths over time + moving average
   - Bottom: Length distribution histogram

3. **losses.png**
   - Q1 Loss (critic network 1)
   - Q2 Loss (critic network 2)
   - Policy Loss (actor network)
   - Alpha Loss (temperature parameter)

4. **alpha.png**
   - Alpha (temperature) value over time
   - Shows how the exploration-exploitation tradeoff evolved

5. **buffer_size.png**
   - Replay buffer size over time
   - Shows when buffer fills up and stabilizes

6. **training_time.png**
   - Training time metrics (total, per episode, per step)

---

## 3. Comparison Results (`results/`)

### Location
`results/` contains files comparing ALL experiments:

### Files

1. **comparison_results.json**
   - Contains evaluation results for each experiment
   - Each entry has:
     - `experiment_name`: Name of the experiment
     - `use_gym_mode`: Whether it used gym mode
     - `step_duration`: Step duration in seconds
     - `checkpoint`: Path to the final checkpoint
     - `episode_rewards`: List of rewards from evaluation episodes
     - `episode_lengths`: List of lengths from evaluation episodes
     - `avg_reward`: Average reward across evaluation episodes
     - `avg_length`: Average episode length
     - `reward_std`: Standard deviation of rewards
     - `max_reward`: Maximum reward achieved

2. **delay_comparison.png**
   - Comprehensive comparison plot with 4 subplots:
     - **Top Left**: Average Reward vs Step Duration (with error bars)
     - **Top Right**: Max Reward vs Step Duration
     - **Bottom Left**: Average Episode Length vs Step Duration
     - **Bottom Right**: Reward Distribution comparison
   - If real-time baseline was included, it shows as a horizontal red dashed line

3. **summary.json**
   - Summary statistics for all experiments

---

## How to Use These Results

### View Individual Training Plots
```bash
# View plots for a specific experiment
ls delay_experiments/experiments_20251218_130600/metrics/gym_mode_0.1s/plots/
# Open any PNG file with an image viewer
```

### Load Training Metrics Programmatically
```python
import json
from pathlib import Path

# Load training metrics
metrics_file = Path("delay_experiments/experiments_20251218_130600/metrics/gym_mode_0.1s/metrics/training_metrics.json")
with open(metrics_file, 'r') as f:
    metrics = json.load(f)

# Access data
episode_rewards = metrics['episode_metrics']['rewards']
avg_reward = metrics['statistics']['episode_rewards']['mean']
```

### Load Comparison Results
```python
import json

# Load comparison results
with open("delay_experiments/experiments_20251218_130600/results/comparison_results.json", 'r') as f:
    results = json.load(f)

# Access results for a specific experiment
for result in results:
    if result['step_duration'] == 0.1:
        print(f"0.1s delay: avg reward = {result['avg_reward']:.2f}")
```

### View Comparison Plot
```bash
# Open the comparison plot
delay_experiments/experiments_20251218_130600/results/delay_comparison.png
```

---

## Quick Reference

| Item | Location |
|------|----------|
| Model checkpoints | `checkpoints/{experiment_name}/` |
| Training metrics JSON | `metrics/{experiment_name}/metrics/training_metrics.json` |
| Training plots | `metrics/{experiment_name}/plots/*.png` |
| Evaluation metrics | `checkpoints/{experiment_name}/evaluation_metrics.json` |
| Comparison plot | `results/delay_comparison.png` |
| Comparison JSON | `results/comparison_results.json` |
| Summary | `results/summary.json` |



