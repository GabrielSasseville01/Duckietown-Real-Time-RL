# Training Metrics System

This system tracks and visualizes training metrics during SAC agent training.

## Features

- **Comprehensive Metrics Tracking**:
  - Episode rewards and lengths
  - Training losses (Q1, Q2, Policy, Alpha)
  - Alpha (temperature) values over time
  - Replay buffer size
  - Training time (total, per episode, per step)
  - Windowed averages for smoothing

- **Automatic Plotting**:
  - Reward plots (raw + moving average + distribution)
  - Episode length plots
  - Loss plots (all networks)
  - Alpha value plots
  - Buffer size plots
  - Training time plots

- **Organized Storage**:
  - Metrics saved as JSON for easy analysis
  - Plots saved as PNG images
  - Organized folder structure

## Folder Structure

```
training_logs/
├── metrics/
│   └── training_metrics.json    # All metrics in JSON format
├── plots/
│   ├── rewards.png               # Episode rewards
│   ├── lengths.png                # Episode lengths
│   ├── losses.png                 # Training losses
│   ├── alpha.png                  # Alpha values
│   ├── buffer_size.png            # Buffer size over time
│   └── training_time.png          # Time metrics
└── checkpoints/                   # (Optional) Checkpoint links
```

## Usage

### During Training

Metrics are automatically tracked and saved during training:

```bash
# Train with metrics (default)
python src/sac_agent.py --num_episodes 1000 --gym_mode

# Train without metrics
python src/sac_agent.py --num_episodes 1000 --no_metrics

# Custom metrics directory
python src/sac_agent.py --num_episodes 1000 --metrics_dir my_training_logs
```

### Plot Metrics After Training

If you have a saved metrics file, you can generate plots:

```bash
# Plot from saved metrics file
python src/plot_metrics.py --metrics_file training_logs/metrics/training_metrics.json

# Plot and display
python src/plot_metrics.py --metrics_file training_logs/metrics/training_metrics.json --show

# Custom save directory
python src/plot_metrics.py --metrics_file training_logs/metrics/training_metrics.json --save_dir my_plots
```

## Tracked Metrics

### Episode-Level Metrics
- **Rewards**: Total reward per episode
- **Lengths**: Number of steps per episode
- **Times**: Time taken per episode
- **Windowed Averages**: Moving average over last 100 episodes

### Step-Level Metrics
- **Rewards**: Reward per step
- **Q1/Q2 Losses**: Critic network losses
- **Policy Loss**: Actor network loss
- **Alpha Loss**: Temperature loss
- **Alpha Values**: Current temperature value
- **Buffer Size**: Replay buffer size
- **Step Times**: Time per step

### Training Statistics
- Total training time
- Mean/std/min/max for all metrics
- Final 100 episode averages

## Example Output

After training, you'll see:
```
Training complete! Model saved to checkpoints/sac_*_final.pth
Metrics saved to: training_logs/metrics/training_metrics.json
Generating training plots...
Plots saved to: training_logs/plots/
Training metrics and plots saved to: training_logs/
```

## Metrics JSON Structure

The saved JSON file contains:
```json
{
  "config": { ... },
  "training_time": { ... },
  "episode_metrics": { ... },
  "step_metrics": { ... },
  "statistics": { ... }
}
```

You can load and analyze this data programmatically:
```python
from training_metrics import load_metrics

metrics = load_metrics("training_logs/metrics/training_metrics.json")
print(metrics["statistics"]["episode_rewards"]["mean"])
```

