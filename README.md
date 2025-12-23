# Duckietown Real-Time Reinforcement Learning

This repository contains implementations of Reinforcement Learning algorithms (SAC, PPO, REINFORCE) for autonomous vehicle control in the Duckietown environment, with support for real-time training and simulation with computation delays.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Delay Experiments](#delay-experiments)
- [Analysis and Visualization](#analysis-and-visualization)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## Features

- **Multiple RL Algorithms**: SAC (Soft Actor-Critic), PPO (Proximal Policy Optimization), REINFORCE
- **Real-Time and Gym Modes**: Train in real-time or with faster gym-mode simulation
- **Computation Delay Simulation**: Study the effects of computation delays on RL performance
- **Action Conditioning**: Support for including previous actions in observations (for real-time RL)
- **Comprehensive Metrics**: Automatic tracking and visualization of training metrics
- **Hyperparameter Tuning**: Automated hyperparameter search with grid search
- **Experiment Management**: Organized experiment tracking with automatic checkpointing

## Prerequisites

- Python 3.8+
- Docker (for Duckietown Matrix simulation)
- CUDA-capable GPU (recommended for faster training)
- Duckietown Shell (`dts`) installed

## Installation

### 1. Clone the Repository

```bash
git clone git@github.com:GabrielSasseville01/Duckietown-Real-Time-RL.git
cd Duckietown-Real-Time-RL
```

### 2. Set Up Virtual Environment

Create a new virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install Duckietown SDK

The Duckietown SDK should be available in the provided environment. If not, follow the [Duckietown SDK installation instructions](https://docs.duckietown.org/daffy/opmanual_duckiebot/out/install_duckietown_shell.html).

### 4. Build Duckietown Matrix

Build the Duckietown Matrix image:

```bash
cd dt-duckiematrix
dts devel build -f
cd ..
```

## Quick Start

### 1. Launch the Matrix Engine

From the project root directory, launch the Duckietown Matrix in gym mode:

```bash
dts matrix run --image duckietown/dt-duckiematrix:ente-amd64 --standalone --map ./dt-duckiematrix/assets/embedded_maps/loop --mode gym --no-pull
```

**Note**: Keep this terminal running! The matrix engine must be running for training/evaluation to work.

**Troubleshooting**: If you get a container conflict error, remove the existing container:
```bash
docker rm -f dts-matrix-engine
```

### 2. Train a SAC Agent (Simple Example)

In a new terminal (with the virtual environment activated):

```bash
cd gym-duckiematrix
python src/sac_agent.py --num_episodes 200 --gym_mode --step_duration 0.1
```

This will:
- Train a SAC agent for 200 episodes
- Use gym mode (faster simulation)
- Save checkpoints every 50 episodes (default)
- Save training metrics and plots to `training_logs/`
- Save model checkpoints to `checkpoints/`

## Training

### Basic Training Command

#### SAC (Recommended)

```bash
python src/sac_agent.py \
    --num_episodes 1000 \
    --max_steps_per_episode 2000 \
    --batch_size 256 \
    --save_freq 100 \
    --checkpoint_dir checkpoints \
    --gym_mode \
    --step_duration 0.1
```

**Key Arguments**:
- `--num_episodes`: Number of training episodes (default: 1500)
- `--max_steps_per_episode`: Maximum steps per episode (default: 2000)
- `--batch_size`: Batch size for network updates (default: 256)
- `--save_freq`: Save checkpoint every N episodes (default: 50)
- `--checkpoint_dir`: Directory to save checkpoints (default: "checkpoints")
- `--gym_mode`: Use gym mode (faster, non-real-time)
- `--step_duration`: Step duration in seconds for gym mode (default: 0.1)
- `--condition_on_prev_action`: Include previous action in observations (for real-time RL)

#### Real-Time Training (Slower)

```bash
python src/sac_agent.py \
    --num_episodes 500 \
    --max_steps_per_episode 2000 \
    --checkpoint_dir checkpoints/realtime
```

(No `--gym_mode` flag = real-time mode)

### Resume Training

To resume from a checkpoint:

```bash
python src/sac_agent.py \
    --num_episodes 1000 \
    --start_episode 200 \
    --policy_checkpoint checkpoints/sac_policy_ep200.pth \
    --q1_checkpoint checkpoints/sac_q1_ep200.pth \
    --q2_checkpoint checkpoints/sac_q2_ep200.pth \
    --checkpoint_dir checkpoints \
    --gym_mode
```

### Training Output

When training, the following files are created:

```
checkpoints/
├── sac_policy_ep100.pth      # Periodic checkpoints
├── sac_q1_ep100.pth
├── sac_q2_ep100.pth
├── sac_policy_ep200.pth
├── ...
├── sac_policy_final.pth       # Final checkpoint
├── sac_q1_final.pth
└── sac_q2_final.pth

training_logs/
├── metrics/
│   └── training_metrics.json  # All training data
└── plots/
    ├── rewards.png            # Reward plots
    ├── lengths.png            # Episode length plots
    ├── losses.png             # Loss plots
    ├── alpha.png              # Temperature parameter
    ├── buffer_size.png        # Replay buffer size
    └── training_time.png      # Time metrics
```

### Custom Hyperparameters

Create a JSON file with hyperparameters:

```json
{
    "lr": 0.0003,
    "gamma": 0.99,
    "tau": 0.01,
    "alpha": 0.2,
    "auto_alpha": true,
    "hidden_dim": 256
}
```

Then use it:

```bash
python src/sac_agent.py \
    --num_episodes 1000 \
    --hyperparams_file hyperparams.json \
    --gym_mode
```

## Evaluation

### Evaluate a Trained Model

```bash
python src/sac_inference.py \
    --policy_checkpoint checkpoints/sac_policy_final.pth \
    --num_episodes 20 \
    --max_steps 2000 \
    --gym_mode \
    --step_duration 0.1 \
    --save_metrics \
    --no_render
```

**Key Arguments**:
- `--policy_checkpoint`: Path to policy checkpoint (required)
- `--q1_checkpoint`, `--q2_checkpoint`: Optional Q-network checkpoints
- `--num_episodes`: Number of evaluation episodes (default: 10)
- `--max_steps`: Maximum steps per episode (default: 2000)
- `--gym_mode`: Use gym mode for faster evaluation
- `--save_metrics`: Save evaluation metrics to JSON
- `--no_render`: Disable rendering delay (faster)
- `--condition_on_prev_action`: Use if model was trained with action conditioning

### Evaluation Output

Evaluation saves metrics to:
```
checkpoints/evaluation_metrics.json
```

Contains:
- Average reward and standard deviation
- Average episode length
- All episode rewards and lengths
- Max/min rewards

## Hyperparameter Tuning

Automated hyperparameter tuning with grid search:

```bash
python src/hyperparameter_tuning.py \
    --num_episodes 400 \
    --max_steps_per_episode 2000 \
    --save_freq 100 \
    --gym_mode \
    --step_duration 0.1 \
    --base_dir hyperparameter_tuning
```

### Resume Hyperparameter Tuning

```bash
python src/hyperparameter_tuning.py \
    --resume_from config_5_lr_0.001_batch_size_512 \
    --base_dir hyperparameter_tuning
```

### Run Specific Configurations

```bash
python src/hyperparameter_tuning.py \
    --only config_0_lr_0.0003_batch_size_256 config_1_lr_0.001_batch_size_256 \
    --base_dir hyperparameter_tuning
```

### Analyze Results

```bash
python src/analyze_hyperparameter_tuning.py \
    --base_dir hyperparameter_tuning
```

Output:
- `hyperparameter_tuning/hyperparameter_tuning_summary.csv`
- `hyperparameter_tuning/analysis_plots/` with various analysis plots

## Delay Experiments

Run comprehensive delay comparison experiments to study the effect of computation delays:

### Basic Delay Experiment

```bash
python src/run_delay_experiments.py \
    --step_durations 0.01 0.033 0.05 0.1 0.5 1.0 \
    --num_episodes 400 \
    --eval_episodes 20 \
    --save_freq 100 \
    --base_dir delay_experiments
```

**Arguments**:
- `--step_durations`: List of step durations to test (in seconds)
- `--num_episodes`: Training episodes per experiment (default: 400)
- `--eval_episodes`: Evaluation episodes (default: 20)
- `--realtime`: Include real-time baseline (default: False)
- `--condition_on_prev_action`: Enable action conditioning
- `--save_freq`: Checkpoint save frequency (default: 100)

### Resume Delay Experiments

If training was interrupted, resume from existing experiment:

```bash
python src/run_delay_experiments.py \
    --resume_experiment delay_experiments/experiments_20251218_130600 \
    --step_durations 0.01 0.033 0.05 0.1 0.5 1.0 \
    --num_episodes 200 \
    --condition_on_prev_action
```

The script will:
- Skip already-completed experiments
- Resume incomplete experiments
- Run evaluation for all experiments
- Update comparison results

### Delay Experiment Output Structure

```
delay_experiments/
└── experiments_YYYYMMDD_HHMMSS/
    ├── checkpoints/
    │   ├── gym_mode_0.01s/
    │   │   ├── sac_policy_ep100.pth
    │   │   ├── sac_policy_ep200.pth
    │   │   ├── sac_policy_final.pth
    │   │   ├── evaluation_metrics.json
    │   │   └── ...
    │   ├── gym_mode_0.033s/
    │   └── ...
    ├── metrics/
    │   ├── gym_mode_0.01s/
    │   │   ├── metrics/training_metrics.json
    │   │   └── plots/*.png
    │   └── ...
    └── results/
        ├── comparison_results.json
        ├── delay_comparison.png
        └── summary.json
```

## Analysis and Visualization

### Analyze Delay Experiments

Generate comprehensive analysis plots from delay experiments:

```bash
python src/analyze_delay_experiments.py \
    --experiment_dir delay_experiments/experiments_YYYYMMDD_HHMMSS \
    --output_dir delay_experiments/experiments_YYYYMMDD_HHMMSS/analysis
```

**Output Files**:
- `reward_vs_delay.png`: Reward analysis across delays
- `episode_length_vs_delay.png`: Episode length comparison
- `learning_curves.png`: Training progress for all delays
- `training_time_analysis.png`: Training efficiency metrics
- `loss_convergence.png`: Loss convergence plots
- `performance_distributions.png`: Performance variability analysis
- `summary_table.png/.csv/.txt`: Summary statistics table

### Compare Experiments

Compare results from multiple experiments:

```bash
python src/compare_experiments.py \
    --results_file delay_experiments/experiments_YYYYMMDD_HHMMSS/results/comparison_results.json
```

### Plot Training Metrics

Generate plots from saved training metrics:

```bash
python src/plot_metrics.py \
    --metrics_file training_logs/metrics/training_metrics.json \
    --save_dir training_logs/plots
```

## Project Structure

```
gym-duckiematrix/
├── src/
│   ├── sac_agent.py              # SAC training script
│   ├── sac_inference.py          # Model evaluation
│   ├── ppo_agent.py              # PPO implementation
│   ├── reinforce_agent.py        # REINFORCE implementation
│   ├── run_delay_experiments.py  # Delay experiment runner
│   ├── hyperparameter_tuning.py  # Hyperparameter search
│   ├── analyze_delay_experiments.py  # Analysis tools
│   ├── compare_experiments.py    # Comparison utilities
│   ├── plot_metrics.py           # Plotting utilities
│   ├── training_metrics.py       # Metrics tracking
│   └── gym_duckiematrix/         # Duckietown environment
├── checkpoints/                  # Default checkpoint directory
├── training_logs/                # Default metrics directory
├── delay_experiments/            # Delay experiment results
├── hyperparameter_tuning/        # Hyperparameter tuning results
├── dt-duckiematrix/              # Duckietown Matrix (submodule)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Troubleshooting

### Matrix Engine Won't Start

**Error**: `Conflict. The container name "/dts-matrix-engine" is already in use`

**Solution**:
```bash
docker rm -f dts-matrix-engine
```

Then try launching again.

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'gymnasium'`

**Solution**: Make sure virtual environment is activated:
```bash
source /path/to/venv/bin/activate
pip install -r requirements.txt
```

### Checkpoint Loading Errors

**Error**: `size mismatch for fc1.weight: copying a param with shape torch.Size([256, 5]) from checkpoint, the shape in current model is torch.Size([256, 3])`

**Solution**: This happens when there's a mismatch in observation dimensions (action conditioning). Make sure to:
- Use `--condition_on_prev_action` if the model was trained with it
- Or omit the flag if it wasn't
- The `run_delay_experiments.py` script auto-detects this

### Out of Memory Errors

**Solution**:
- Reduce `--batch_size` (default: 256)
- Reduce `--max_steps_per_episode`
- Use gym mode (`--gym_mode`) instead of real-time

### Training Too Slow

**Solutions**:
- Use `--gym_mode` flag for faster simulation
- Reduce `--step_duration` in gym mode (e.g., 0.01s)
- Use GPU (automatically detected)
- Reduce `--max_steps_per_episode`

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{duckietown-realtime-rl,
  title={Duckietown Real-Time Reinforcement Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/repo}
}
```

## License

[Specify your license here]

## Acknowledgments

- Duckietown project for the simulation environment
- Original SAC implementation inspiration
