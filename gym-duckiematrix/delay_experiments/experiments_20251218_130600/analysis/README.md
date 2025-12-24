# Delay Experiment Analysis

This directory contains comprehensive analysis plots and tables generated from all delay experiments.

## Generated Files

### 1. **reward_vs_delay.png**
Four-panel comparison of rewards:
- **Top Left**: Training reward (final 50 episodes average) vs step duration
- **Top Right**: Evaluation reward vs step duration  
- **Bottom Left**: Maximum training reward vs step duration
- **Bottom Right**: Evaluation reward stability (std dev) - lower is more stable

### 2. **episode_length_vs_delay.png**
Two-panel comparison of episode lengths:
- **Left**: Training episode length (final 50 episodes) vs step duration
- **Right**: Evaluation episode length vs step duration

### 3. **learning_curves.png**
Learning curves showing training progress:
- **Top**: Episode rewards over training (with smoothing) for all delays
- **Bottom**: Episode lengths over training (with smoothing) for all delays
- Helps visualize how quickly each configuration learns

### 4. **training_time_analysis.png**
Four-panel training time analysis:
- **Top Left**: Total training time (hours) vs step duration
- **Top Right**: Time per episode (seconds) vs step duration
- **Bottom Left**: Sample efficiency - episodes needed to reach a reward threshold
- **Bottom Right**: Training time vs final performance (scatter plot colored by delay)

### 5. **loss_convergence.png**
Loss convergence for all network components:
- **Top Left**: Q1 Loss (Critic Network 1)
- **Top Right**: Q2 Loss (Critic Network 2)
- **Bottom Left**: Policy Loss (Actor Network)
- **Bottom Right**: Alpha Loss (Temperature parameter)
- All plots use log scale for better visualization

### 6. **performance_distributions.png**
Performance distribution analysis:
- **Top Left**: Training reward distributions (final 100 episodes) - histograms
- **Top Right**: Evaluation reward distributions - histograms
- **Bottom Left**: Training reward box plots by delay
- **Bottom Right**: Evaluation reward box plots by delay
- Shows variance and stability of performance

### 7. **summary_table.png / .csv / .txt**
Comprehensive summary statistics table with:
- Step duration
- Number of training episodes
- Final average reward (training)
- Maximum reward (training)
- Final average episode length (training)
- Evaluation average reward
- Evaluation reward standard deviation
- Evaluation average episode length
- Total training time (hours)

## Key Insights to Look For

1. **Performance vs Delay**: How does step duration affect final performance?
2. **Training Time**: Which delays train fastest/slowest?
3. **Sample Efficiency**: Which delays require fewer episodes to learn?
4. **Stability**: Which delays have more consistent/reliable performance?
5. **Loss Convergence**: Do all delays converge similarly, or do some struggle?

## Usage

To regenerate these plots with updated data:

```bash
python src/analyze_delay_experiments.py --experiment_dir delay_experiments/experiments_20251218_130600
```

Or for a different experiment directory:

```bash
python src/analyze_delay_experiments.py --experiment_dir path/to/experiment --output_dir path/to/output
```



