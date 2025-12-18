# Hyperparameter Tuning Guide

This guide explains how to use the hyperparameter tuning system for the SAC agent.

## Overview

The hyperparameter tuning system performs a grid search over key hyperparameters, training each configuration for a specified number of episodes. It supports:
- **Resuming**: Continue from a specific configuration if training is interrupted
- **Selective runs**: Run only specific configurations
- **Progress tracking**: Automatically tracks which configurations are completed
- **Results analysis**: Analyze and visualize results after tuning

## Hyperparameter Grid

The default grid searches over 4 key hyperparameters:

| Hyperparameter | Values | Description |
|---------------|--------|-------------|
| `lr` | [1e-4, 3e-4] | Learning rate |
| `hidden_dim` | [128, 256] | Network hidden layer dimension |
| `tau` | [0.005, 0.01] | Soft update coefficient for target networks |
| `alpha` | [0.1, 0.2] | Initial entropy regularization coefficient |

**Total configurations**: 2 × 2 × 2 × 2 = **16 configurations**

Note: Other hyperparameters use fixed defaults (`batch_size=256`, `update_freq=1`).

## Usage

### Basic Usage

Run all configurations with default settings (400 episodes each):

```bash
python src/hyperparameter_tuning.py
```

### Custom Settings

```bash
python src/hyperparameter_tuning.py \
    --num_episodes 400 \
    --max_steps_per_episode 2000 \
    --save_freq 100 \
    --gym_mode \
    --step_duration 0.1 \
    --base_dir hyperparameter_tuning
```

### Resuming from a Specific Configuration

If training is interrupted, you can resume from a specific configuration:

```bash
python src/hyperparameter_tuning.py --resume_from config_5_lr_3e-04_batch_size_256_hidden_dim_256_tau_0p005_alpha_0p2_update_freq_1
```

The script will automatically detect the latest checkpoint for that configuration and resume from there.

### Running Only Specific Configurations

To run only a subset of configurations:

```bash
python src/hyperparameter_tuning.py --only config_0 config_5 config_10
```

### Skipping Completed Configurations

By default, the script skips configurations that have already completed. To disable this:

```bash
python src/hyperparameter_tuning.py --no_skip_completed
```

## Directory Structure

Each hyperparameter tuning run creates a directory structure like:

```
hyperparameter_tuning/
└── tuning_20251212_120000/
    ├── hyperparameter_grid.json          # Grid definition
    ├── progress.json                     # Progress tracking
    ├── config_0_lr_1e-04_batch_size_128_.../
    │   ├── checkpoints/
    │   │   ├── sac_policy_ep100.pth
    │   │   ├── sac_q1_ep100.pth
    │   │   ├── sac_q2_ep100.pth
    │   │   └── ...
    │   └── training_logs/
    │       ├── training_metrics.json
    │       └── ...
    ├── config_1_.../
    └── ...
```

## Analyzing Results

After training completes (or partially completes), analyze the results:

```bash
python src/analyze_hyperparameter_tuning.py --experiment_dir hyperparameter_tuning/tuning_20251212_120000
```

This generates:
1. **Summary table**: CSV file with all configurations and their performance
2. **Analysis plots**: 
   - Performance vs each hyperparameter
   - Learning curves for top 5 configurations
3. **Best configuration**: Identifies the best performing configuration

### Output Files

- `hyperparameter_tuning_summary.csv`: Summary table with all results
- `analysis_plots/hyperparameter_analysis.png`: Performance vs hyperparameters
- `analysis_plots/top5_learning_curves.png`: Learning curves for top configurations

## Customizing the Grid

To customize the hyperparameter grid, edit `generate_hyperparameter_grid()` in `src/hyperparameter_tuning.py`:

```python
def generate_hyperparameter_grid():
    hyperparams = {
        'lr': [1e-4, 3e-4, 1e-3],  # Add more values
        'batch_size': [128, 256, 512],
        # ... modify as needed
    }
    # ...
```

## Tips

1. **Start small**: Test with a few configurations first using `--only`
2. **Monitor progress**: Check `progress.json` to see which configurations are completed
3. **Resume capability**: The script automatically resumes from the latest checkpoint
4. **Gym mode**: Use `--gym_mode` for faster training (recommended)
5. **Checkpoint frequency**: Use `--save_freq 100` to save checkpoints every 100 episodes

## Example Workflow

1. **Start tuning**:
   ```bash
   python src/hyperparameter_tuning.py --num_episodes 400 --gym_mode
   ```

2. **If interrupted, resume**:
   ```bash
   # Check progress.json to see where it stopped
   python src/hyperparameter_tuning.py --resume_from config_42_...
   ```

3. **Analyze results**:
   ```bash
   python src/analyze_hyperparameter_tuning.py --experiment_dir hyperparameter_tuning/tuning_20251212_120000
   ```

4. **Use best configuration**:
   - Check the summary CSV for the best configuration
   - Use those hyperparameters for final training

## Troubleshooting

- **Out of memory**: Reduce `batch_size` or `hidden_dim` in the grid
- **Too slow**: Use `--gym_mode` and reduce `num_episodes` for initial exploration
- **Config not found**: Check the exact config name in `progress.json`
- **Missing metrics**: Ensure training completed successfully for that configuration

