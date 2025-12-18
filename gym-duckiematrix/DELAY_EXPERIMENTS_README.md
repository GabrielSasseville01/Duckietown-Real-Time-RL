# Delay Experiments - Comparing Real-time vs Gym Mode Performance

This system allows you to systematically compare SAC agent performance between real-time mode and gym mode with varying step durations (simulating different delay values).

## Overview

The goal is to measure how performance degrades as step duration increases, simulating higher delays in the system.

## Quick Start

Run a complete experiment suite:

```bash
# Run experiments with default settings
python src/run_delay_experiments.py

# Custom step durations and training episodes
python src/run_delay_experiments.py \
    --step_durations 0.05 0.1 0.2 0.5 1.0 \
    --num_episodes 500 \
    --eval_episodes 20
```

## What It Does

1. **Trains multiple agents**:
   - One real-time baseline agent
   - Multiple gym mode agents with different step durations (e.g., 0.05s, 0.1s, 0.2s, 0.5s, 1.0s)

2. **Evaluates each agent**:
   - Runs evaluation episodes
   - Collects performance metrics

3. **Generates comparison plots**:
   - Performance vs step duration
   - Performance degradation relative to real-time
   - Summary statistics

## Folder Structure

```
delay_experiments/
└── experiments_YYYYMMDD_HHMMSS/
    ├── checkpoints/
    │   ├── realtime_baseline/
    │   │   └── sac_policy_final.pth
    │   ├── gym_mode_0.05s/
    │   │   └── sac_policy_final.pth
    │   ├── gym_mode_0.1s/
    │   │   └── sac_policy_final.pth
    │   └── ...
    ├── metrics/
    │   ├── realtime_baseline/
    │   │   └── training_metrics.json
    │   └── ...
    └── results/
        ├── comparison_results.json
        ├── summary.json
        └── delay_comparison.png
```

## Command Line Options

```bash
python src/run_delay_experiments.py \
    --step_durations 0.05 0.1 0.2 0.5 1.0 \    # Step durations to test
    --num_episodes 500 \                        # Training episodes per experiment
    --max_steps_per_episode 2000 \             # Max steps during training
    --eval_episodes 20 \                        # Evaluation episodes
    --eval_max_steps 2000 \                     # Max steps during evaluation
    --base_dir delay_experiments \              # Base directory
    --no_realtime \                             # Skip real-time baseline
    --batch_size 256 \                          # Training batch size
    --update_freq 1 \                           # Update frequency
    --save_freq 50                               # Save frequency
```

## Analyzing Results

### View Comparison Plot

The script automatically generates `delay_comparison.png` with:
- Average reward vs step duration
- Max reward vs step duration
- Episode length vs step duration
- Performance degradation percentage

### View Summary

```bash
# View summary JSON
cat delay_experiments/experiments_*/results/summary.json

# Or use the comparison script directly
python src/compare_experiments.py \
    --results_file delay_experiments/experiments_*/results/comparison_results.json
```

### Manual Comparison

If you want to compare results from different experiment runs:

```bash
python src/compare_experiments.py \
    --results_file path/to/comparison_results.json \
    --save_dir path/to/output
```

## Expected Results

You should see:
- **Real-time baseline**: Best performance (no delay)
- **Gym mode with small delays (0.05-0.1s)**: Near baseline performance
- **Gym mode with larger delays (0.5-1.0s)**: Degraded performance
- **Performance degradation**: Increasing percentage as delay increases

## Tips

1. **Start with fewer episodes** for quick testing:
   ```bash
   python src/run_delay_experiments.py --num_episodes 100 --eval_episodes 5
   ```

2. **Use fewer step durations** for faster experiments:
   ```bash
   python src/run_delay_experiments.py --step_durations 0.1 0.5 1.0
   ```

3. **Skip real-time baseline** if you only want gym mode comparisons:
   ```bash
   python src/run_delay_experiments.py --no_realtime
   ```

4. **Resume from checkpoints**: Each experiment saves checkpoints, so you can resume training if needed.

## Output Files

- **comparison_results.json**: All evaluation metrics for all experiments
- **summary.json**: Summary with performance degradation percentages
- **delay_comparison.png**: Visual comparison plots
- **evaluation_metrics.json**: Per-experiment evaluation details (in each checkpoint folder)

