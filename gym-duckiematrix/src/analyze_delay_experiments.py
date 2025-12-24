"""
Comprehensive analysis script for delay experiments.
Loads all metrics and generates analysis plots and tables.
"""

import json
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from collections import defaultdict


def load_training_metrics(metrics_dir: Path) -> Optional[Dict]:
    """Load training metrics JSON file."""
    metrics_file = metrics_dir / "metrics" / "training_metrics.json"
    if not metrics_file.exists():
        return None
    with open(metrics_file, 'r') as f:
        return json.load(f)


def infer_episodes_from_checkpoints(checkpoints_dir: Path, exp_name: str) -> Optional[int]:
    """Infer number of training episodes from checkpoint filenames when metrics file is missing."""
    checkpoint_dir = checkpoints_dir / exp_name
    if not checkpoint_dir.exists():
        return None
    
    # Look for checkpoint files matching pattern sac_policy_ep{num}.pth or sac_policy_final.pth
    max_episode = 0
    for file in checkpoint_dir.glob("sac_policy_*.pth"):
        match = re.search(r'ep(\d+)', file.name)
        if match:
            episode = int(match.group(1))
            max_episode = max(max_episode, episode)
        elif 'final' in file.name:
            # If final checkpoint exists, training completed, but we need to find the episode count
            # Check other checkpoint files
            pass
    
    # If we found checkpoints but no episode numbers, check if final exists
    # This suggests training completed, but we can't determine exact episode count
    if max_episode > 0:
        return max_episode
    
    # If final checkpoint exists but no numbered checkpoints, we can't determine episodes
    final_checkpoint = checkpoint_dir / "sac_policy_final.pth"
    if final_checkpoint.exists():
        # Return None to indicate training happened but we don't know episode count
        return None
    
    return None


def load_all_experiments(experiment_dir: Path) -> Dict:
    """
    Load all experiment data.
    
    Returns:
        Dictionary with experiment data keyed by step_duration
    """
    results = {}
    metrics_dir = experiment_dir / "metrics"
    checkpoints_dir = experiment_dir / "checkpoints"
    
    # Load comparison results (evaluation metrics) from comparison_results.json
    comparison_file = experiment_dir / "results" / "comparison_results.json"
    comparison_results = {}
    if comparison_file.exists():
        with open(comparison_file, 'r') as f:
            eval_data = json.load(f)
            for exp in eval_data:
                if exp.get("use_gym_mode", False):
                    delay = exp["step_duration"]
                    comparison_results[delay] = exp
    
    # Also check checkpoints directory for experiments that might not have metrics directories
    all_exp_dirs = set()
    if metrics_dir.exists():
        for exp_dir in metrics_dir.iterdir():
            if exp_dir.is_dir():
                all_exp_dirs.add(exp_dir.name)
    if checkpoints_dir.exists():
        for exp_dir in checkpoints_dir.iterdir():
            if exp_dir.is_dir() and exp_dir.name.startswith("gym_mode_"):
                all_exp_dirs.add(exp_dir.name)
    
    # Load training metrics for each experiment
    for exp_dir_name in sorted(all_exp_dirs):
        # Extract step duration from directory name (e.g., "gym_mode_0.1s" -> 0.1)
        try:
            delay_str = exp_dir_name.replace("gym_mode_", "").replace("s", "")
            delay = float(delay_str)
        except (ValueError, AttributeError):
            continue
        
        # Try to load training metrics (may not exist)
        exp_dir = metrics_dir / exp_dir_name if metrics_dir.exists() else None
        training_metrics = load_training_metrics(exp_dir) if exp_dir and exp_dir.exists() else None
        
        # Try to load evaluation metrics from comparison_results.json first
        eval_data = comparison_results.get(delay, None)
        
        # If not found in comparison_results.json, try loading from individual evaluation_metrics.json
        if eval_data is None:
            eval_metrics_file = checkpoints_dir / exp_dir_name / "evaluation_metrics.json"
            if eval_metrics_file.exists():
                try:
                    with open(eval_metrics_file, 'r') as f:
                        eval_metrics = json.load(f)
                    # Extract step duration from training config or use delay from directory name
                    step_duration = delay
                    if training_metrics and "config" in training_metrics:
                        step_duration = training_metrics["config"].get("step_duration", delay)
                    
                    # Format like comparison_results.json
                    eval_data = {
                        "experiment_name": exp_dir_name,
                        "use_gym_mode": True,
                        "step_duration": step_duration,
                        "checkpoint": str(checkpoints_dir / exp_dir_name / "sac_policy_final.pth"),
                        **eval_metrics
                    }
                except Exception as e:
                    print(f"Warning: Could not load evaluation metrics from {eval_metrics_file}: {e}")
                    eval_data = None
        
        # Only skip if we have neither training nor evaluation metrics
        if training_metrics is None and eval_data is None:
            continue
        
        # If training metrics are missing but checkpoints exist, try to infer episode count
        inferred_episodes = None
        if training_metrics is None and checkpoints_dir.exists():
            inferred_episodes = infer_episodes_from_checkpoints(checkpoints_dir, exp_dir_name)
        
        results[delay] = {
            "training": training_metrics,
            "evaluation": eval_data,
            "name": exp_dir_name,
            "inferred_episodes": inferred_episodes
        }
    
    return results


def plot_reward_vs_delay(data: Dict, save_dir: Path):
    """Plot reward vs delay for both training and evaluation."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    delays = sorted(data.keys())
    
    # Training: Final average reward
    ax = axes[0, 0]
    training_rewards = []
    training_reward_stds = []
    
    for delay in delays:
        training = data[delay]["training"]
        if training and "episode_metrics" in training:
            rewards = training["episode_metrics"]["rewards"]
            if rewards:
                # Use last 50 episodes for final performance
                final_rewards = rewards[-50:] if len(rewards) >= 50 else rewards
                training_rewards.append(np.mean(final_rewards))
                training_reward_stds.append(np.std(final_rewards))
            else:
                training_rewards.append(np.nan)
                training_reward_stds.append(np.nan)
        else:
            training_rewards.append(np.nan)
            training_reward_stds.append(np.nan)
    
    ax.errorbar(delays, training_rewards, yerr=training_reward_stds,
                marker='o', linestyle='-', linewidth=2, markersize=8,
                capsize=5, label='Training (final 50 episodes)')
    ax.set_xlabel('Step Duration (seconds)', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Training Reward vs Step Duration', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Evaluation: Average reward
    ax = axes[0, 1]
    eval_rewards = []
    eval_reward_stds = []
    
    for delay in delays:
        eval_data = data[delay]["evaluation"]
        if eval_data and "avg_reward" in eval_data:
            eval_rewards.append(eval_data["avg_reward"])
            eval_reward_stds.append(eval_data.get("reward_std", 0))
        else:
            eval_rewards.append(np.nan)
            eval_reward_stds.append(0)
    
    ax.errorbar(delays, eval_rewards, yerr=eval_reward_stds,
                marker='s', linestyle='--', linewidth=2, markersize=8,
                capsize=5, color='green', label='Evaluation')
    ax.set_xlabel('Step Duration (seconds)', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Evaluation Reward vs Step Duration', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Training: Max reward
    ax = axes[1, 0]
    max_rewards = []
    for delay in delays:
        training = data[delay]["training"]
        if training and "episode_metrics" in training:
            rewards = training["episode_metrics"]["rewards"]
            if rewards:
                max_rewards.append(np.max(rewards))
            else:
                max_rewards.append(np.nan)
        else:
            max_rewards.append(np.nan)
    
    ax.plot(delays, max_rewards, marker='^', linestyle='-', linewidth=2,
            markersize=8, color='orange', label='Max Training Reward')
    ax.set_xlabel('Step Duration (seconds)', fontsize=12)
    ax.set_ylabel('Max Reward', fontsize=12)
    ax.set_title('Best Training Performance vs Step Duration', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Evaluation: Reward stability (std)
    ax = axes[1, 1]
    reward_stability = []
    for delay in delays:
        eval_data = data[delay]["evaluation"]
        if eval_data and "reward_std" in eval_data:
            reward_stability.append(eval_data["reward_std"])
        else:
            reward_stability.append(np.nan)
    
    ax.bar(range(len(delays)), reward_stability, color='purple', alpha=0.7)
    ax.set_xticks(range(len(delays)))
    ax.set_xticklabels([f'{d:.3f}' for d in delays], rotation=45)
    ax.set_xlabel('Step Duration (seconds)', fontsize=12)
    ax.set_ylabel('Reward Std Dev', fontsize=12)
    ax.set_title('Evaluation Reward Stability (Lower = More Stable)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / "reward_vs_delay.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_episode_length_vs_delay(data: Dict, save_dir: Path):
    """Plot episode length vs delay."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    delays = sorted(data.keys())
    
    # Training: Average episode length
    ax = axes[0]
    training_lengths = []
    training_length_stds = []
    
    for delay in delays:
        training = data[delay]["training"]
        if training and "episode_metrics" in training:
            lengths = training["episode_metrics"]["lengths"]
            if lengths:
                final_lengths = lengths[-50:] if len(lengths) >= 50 else lengths
                training_lengths.append(np.mean(final_lengths))
                training_length_stds.append(np.std(final_lengths))
            else:
                training_lengths.append(np.nan)
                training_length_stds.append(0)
        else:
            training_lengths.append(np.nan)
            training_length_stds.append(0)
    
    ax.errorbar(delays, training_lengths, yerr=training_length_stds,
                marker='o', linestyle='-', linewidth=2, markersize=8,
                capsize=5, label='Training (final 50 episodes)')
    ax.set_xlabel('Step Duration (seconds)', fontsize=12)
    ax.set_ylabel('Average Episode Length (steps)', fontsize=12)
    ax.set_title('Training Episode Length vs Step Duration', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Evaluation: Average episode length
    ax = axes[1]
    eval_lengths = []
    for delay in delays:
        eval_data = data[delay]["evaluation"]
        if eval_data and "avg_length" in eval_data:
            eval_lengths.append(eval_data["avg_length"])
        else:
            eval_lengths.append(np.nan)
    
    ax.plot(delays, eval_lengths, marker='s', linestyle='--', linewidth=2,
            markersize=8, color='green', label='Evaluation')
    ax.set_xlabel('Step Duration (seconds)', fontsize=12)
    ax.set_ylabel('Average Episode Length (steps)', fontsize=12)
    ax.set_title('Evaluation Episode Length vs Step Duration', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(save_dir / "episode_length_vs_delay.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_learning_curves(data: Dict, save_dir: Path):
    """Plot learning curves for all experiments."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    delays = sorted(data.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(delays)))
    
    # Rewards over training
    ax = axes[0]
    for delay, color in zip(delays, colors):
        training = data[delay]["training"]
        if training and "episode_metrics" in training:
            rewards = training["episode_metrics"]["rewards"]
            if rewards:
                # Smooth with moving average
                window = min(20, len(rewards) // 10)
                if window > 1:
                    smoothed = pd.Series(rewards).rolling(window=window, center=True).mean()
                    ax.plot(rewards, alpha=0.2, color=color, linewidth=0.5)
                    ax.plot(smoothed, label=f'{delay:.3f}s', color=color, linewidth=2)
                else:
                    ax.plot(rewards, label=f'{delay:.3f}s', color=color, linewidth=1.5)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Learning Curves: Reward Over Training', fontsize=14, fontweight='bold')
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Episode lengths over training
    ax = axes[1]
    for delay, color in zip(delays, colors):
        training = data[delay]["training"]
        if training and "episode_metrics" in training:
            lengths = training["episode_metrics"]["lengths"]
            if lengths:
                window = min(20, len(lengths) // 10)
                if window > 1:
                    smoothed = pd.Series(lengths).rolling(window=window, center=True).mean()
                    ax.plot(lengths, alpha=0.2, color=color, linewidth=0.5)
                    ax.plot(smoothed, label=f'{delay:.3f}s', color=color, linewidth=2)
                else:
                    ax.plot(lengths, label=f'{delay:.3f}s', color=color, linewidth=1.5)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Length (steps)', fontsize=12)
    ax.set_title('Learning Curves: Episode Length Over Training', fontsize=14, fontweight='bold')
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "learning_curves.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_time_analysis(data: Dict, save_dir: Path):
    """Plot training time analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    delays = sorted(data.keys())
    
    # Total training time
    ax = axes[0, 0]
    total_times = []
    for delay in delays:
        training = data[delay]["training"]
        if training and "training_time" in training:
            total_hours = training["training_time"].get("total_hours", 0)
            total_times.append(total_hours)
        else:
            total_times.append(np.nan)
    
    ax.bar(range(len(delays)), total_times, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(delays)))
    ax.set_xticklabels([f'{d:.3f}' for d in delays], rotation=45)
    ax.set_xlabel('Step Duration (seconds)', fontsize=12)
    ax.set_ylabel('Total Training Time (hours)', fontsize=12)
    ax.set_title('Total Training Time vs Step Duration', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Time per episode
    ax = axes[0, 1]
    time_per_episode = []
    for delay in delays:
        training = data[delay]["training"]
        if training and "training_time" in training and "episode_metrics" in training:
            total_hours = training["training_time"].get("total_hours", 0)
            num_episodes = len(training["episode_metrics"].get("rewards", []))
            if num_episodes > 0:
                time_per_episode.append(total_hours * 3600 / num_episodes)  # seconds per episode
            else:
                time_per_episode.append(np.nan)
        else:
            time_per_episode.append(np.nan)
    
    ax.bar(range(len(delays)), time_per_episode, color='coral', alpha=0.7)
    ax.set_xticks(range(len(delays)))
    ax.set_xticklabels([f'{d:.3f}' for d in delays], rotation=45)
    ax.set_xlabel('Step Duration (seconds)', fontsize=12)
    ax.set_ylabel('Time per Episode (seconds)', fontsize=12)
    ax.set_title('Training Time per Episode vs Step Duration', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Sample efficiency: episodes to reach threshold
    ax = axes[1, 0]
    threshold = -1.0  # Reward threshold
    episodes_to_threshold = []
    
    for delay in delays:
        training = data[delay]["training"]
        if training and "episode_metrics" in training:
            rewards = training["episode_metrics"]["rewards"]
            if rewards:
                # Find first episode where reward exceeds threshold
                for i, r in enumerate(rewards):
                    if r >= threshold:
                        episodes_to_threshold.append(i + 1)
                        break
                else:
                    episodes_to_threshold.append(len(rewards))  # Never reached
            else:
                episodes_to_threshold.append(np.nan)
        else:
            episodes_to_threshold.append(np.nan)
    
    ax.bar(range(len(delays)), episodes_to_threshold, color='mediumseagreen', alpha=0.7)
    ax.set_xticks(range(len(delays)))
    ax.set_xticklabels([f'{d:.3f}' for d in delays], rotation=45)
    ax.set_xlabel('Step Duration (seconds)', fontsize=12)
    ax.set_ylabel('Episodes to Reach Threshold', fontsize=12)
    ax.set_title(f'Sample Efficiency (Episodes to Reach Reward ≥ {threshold})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Training time vs final performance
    ax = axes[1, 1]
    final_rewards = []
    for delay in delays:
        training = data[delay]["training"]
        if training and "episode_metrics" in training:
            rewards = training["episode_metrics"]["rewards"]
            if rewards:
                final_rewards.append(np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards))
            else:
                final_rewards.append(np.nan)
        else:
            final_rewards.append(np.nan)
    
    scatter = ax.scatter(total_times, final_rewards, s=100, c=delays, cmap='viridis', alpha=0.6)
    for i, delay in enumerate(delays):
        ax.annotate(f'{delay:.3f}s', (total_times[i], final_rewards[i]), 
                   fontsize=9, alpha=0.7)
    ax.set_xlabel('Total Training Time (hours)', fontsize=12)
    ax.set_ylabel('Final Average Reward', fontsize=12)
    ax.set_title('Training Time vs Final Performance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Step Duration (s)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_dir / "training_time_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_loss_convergence(data: Dict, save_dir: Path):
    """Plot loss convergence for all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    delays = sorted(data.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(delays)))
    
    loss_types = [
        ("q1_losses", "Q1 Loss (Critic)", axes[0, 0]),
        ("q2_losses", "Q2 Loss (Critic)", axes[0, 1]),
        ("policy_losses", "Policy Loss |Actor| (Absolute)", axes[1, 0]),
        ("alpha_losses", "Alpha Loss (Temperature)", axes[1, 1])
    ]
    
    for loss_key, loss_name, ax in loss_types:
        for delay, color in zip(delays, colors):
            training = data[delay]["training"]
            if training and "step_metrics" in training:
                losses = training["step_metrics"].get(loss_key, [])
                if losses and len(losses) > 0:
                    # Convert to numpy array and filter out None/NaN values
                    losses_array = np.array(losses)
                    losses_array = losses_array[~np.isnan(losses_array)]
                    
                    # Policy loss can be negative in SAC, so use absolute value for log scale
                    # or use linear scale. Let's use absolute value to keep log scale consistency
                    if loss_key == "policy_losses":
                        # For policy loss, plot absolute value since it's typically negative
                        losses_array = np.abs(losses_array)
                        # Filter out zeros
                        losses_array = losses_array[losses_array > 0]
                    else:
                        # For other losses (Q1, Q2, Alpha), filter out non-positive values
                        losses_array = losses_array[losses_array > 0]
                    
                    if len(losses_array) > 0:
                        # Smooth with moving average
                        window = min(100, len(losses_array) // 10)
                        if window > 1:
                            smoothed = pd.Series(losses_array).rolling(window=window, center=True).mean()
                            # Only plot if we have enough points
                            if len(smoothed.dropna()) > 0:
                                ax.plot(losses_array, alpha=0.2, color=color, linewidth=0.5)
                                ax.plot(smoothed, label=f'{delay:.3f}s', color=color, linewidth=2)
                        else:
                            ax.plot(losses_array, label=f'{delay:.3f}s', color=color, linewidth=1.5)
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(loss_name, fontsize=14, fontweight='bold')
        if ax.get_legend_handles_labels()[0]:  # Only add legend if there are labels
            ax.legend(loc='best', ncol=2, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_dir / "loss_convergence.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_table(data: Dict, save_dir: Path):
    """Create a summary statistics table."""
    delays = sorted(data.keys())
    
    rows = []
    for delay in delays:
        training = data[delay]["training"]
        eval_data = data[delay]["evaluation"]
        
        row = {"Step Duration (s)": f"{delay:.3f}"}
        
        # Training metrics
        if training and "episode_metrics" in training:
            rewards = training["episode_metrics"]["rewards"]
            lengths = training["episode_metrics"]["lengths"]
            if rewards:
                row["Training Episodes"] = len(rewards)
                row["Final Avg Reward"] = f"{np.mean(rewards[-50:]):.2f}" if len(rewards) >= 50 else f"{np.mean(rewards):.2f}"
                row["Max Reward"] = f"{np.max(rewards):.2f}"
                row["Final Avg Length"] = f"{np.mean(lengths[-50:]):.1f}" if len(lengths) >= 50 else f"{np.mean(lengths):.1f}"
            else:
                row["Training Episodes"] = 0
                row["Final Avg Reward"] = "N/A"
                row["Max Reward"] = "N/A"
                row["Final Avg Length"] = "N/A"
        else:
            # Try to infer from checkpoints if available
            inferred_episodes = data[delay].get("inferred_episodes")
            if inferred_episodes is not None:
                row["Training Episodes"] = inferred_episodes
            else:
                row["Training Episodes"] = 0
            row["Final Avg Reward"] = "N/A"
            row["Max Reward"] = "N/A"
            row["Final Avg Length"] = "N/A"
        
        # Evaluation metrics
        if eval_data:
            row["Eval Avg Reward"] = f"{eval_data.get('avg_reward', 0):.2f}"
            row["Eval Reward Std"] = f"{eval_data.get('reward_std', 0):.3f}"
            row["Eval Avg Length"] = f"{eval_data.get('avg_length', 0):.1f}"
        else:
            row["Eval Avg Reward"] = "N/A"
            row["Eval Reward Std"] = "N/A"
            row["Eval Avg Length"] = "N/A"
        
        # Training time
        if training and "training_time" in training:
            row["Training Time (h)"] = f"{training['training_time'].get('total_hours', 0):.2f}"
        else:
            row["Training Time (h)"] = "N/A"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save as CSV
    df.to_csv(save_dir / "summary_table.csv", index=False)
    
    # Save as formatted table (text)
    with open(save_dir / "summary_table.txt", 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("DELAY EXPERIMENT SUMMARY TABLE\n")
        f.write("=" * 120 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
    
    # Create a visual table plot
    fig, ax = plt.subplots(figsize=(16, len(rows) * 0.8 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Experiment Summary Statistics', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_dir / "summary_table.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


def plot_performance_distributions(data: Dict, save_dir: Path):
    """Plot performance distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    delays = sorted(data.keys())
    
    # Training reward distributions
    ax = axes[0, 0]
    for delay in delays:
        training = data[delay]["training"]
        if training and "episode_metrics" in training:
            rewards = training["episode_metrics"]["rewards"]
            if rewards:
                # Use final 100 episodes
                final_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
                ax.hist(final_rewards, alpha=0.5, label=f'{delay:.3f}s', bins=30)
    ax.set_xlabel('Reward', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Training Reward Distributions (Final 100 Episodes)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Evaluation reward distributions
    ax = axes[0, 1]
    for delay in delays:
        eval_data = data[delay]["evaluation"]
        if eval_data and "episode_rewards" in eval_data:
            rewards = eval_data["episode_rewards"]
            if rewards:
                ax.hist(rewards, alpha=0.5, label=f'{delay:.3f}s', bins=20)
    ax.set_xlabel('Reward', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Evaluation Reward Distributions', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Box plot: Training rewards by delay
    ax = axes[1, 0]
    box_data = []
    labels = []
    for delay in delays:
        training = data[delay]["training"]
        if training and "episode_metrics" in training:
            rewards = training["episode_metrics"]["rewards"]
            if rewards:
                final_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
                box_data.append(final_rewards)
                labels.append(f'{delay:.3f}s')
    if box_data:
        bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        ax.set_xlabel('Step Duration (seconds)', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('Training Reward Distribution by Delay (Final 100 Episodes)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    # Box plot: Evaluation rewards by delay
    ax = axes[1, 1]
    box_data = []
    labels = []
    for delay in delays:
        eval_data = data[delay]["evaluation"]
        if eval_data and "episode_rewards" in eval_data:
            rewards = eval_data["episode_rewards"]
            if rewards:
                box_data.append(rewards)
                labels.append(f'{delay:.3f}s')
    if box_data:
        bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightgreen')
            patch.set_alpha(0.7)
        ax.set_xlabel('Step Duration (seconds)', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('Evaluation Reward Distribution by Delay', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_dir / "performance_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze delay experiment results')
    parser.add_argument('--experiment_dir', type=str,
                       default='delay_experiments/experiments_20251218_130600',
                       help='Path to experiment directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for analysis (default: experiment_dir/analysis)')
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        raise ValueError(f"Experiment directory not found: {experiment_dir}")
    
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading experiments from: {experiment_dir}")
    print(f"Saving analysis to: {output_dir}")
    
    # Load all experiment data
    data = load_all_experiments(experiment_dir)
    print(f"Loaded {len(data)} experiments")
    
    if not data:
        print("No experiment data found!")
        return
    
    print("\nGenerating plots...")
    
    # Generate all plots
    print("  - Reward vs Delay...")
    plot_reward_vs_delay(data, output_dir)
    
    print("  - Episode Length vs Delay...")
    plot_episode_length_vs_delay(data, output_dir)
    
    print("  - Learning Curves...")
    plot_learning_curves(data, output_dir)
    
    print("  - Training Time Analysis...")
    plot_training_time_analysis(data, output_dir)
    
    print("  - Loss Convergence...")
    plot_loss_convergence(data, output_dir)
    
    print("  - Performance Distributions...")
    plot_performance_distributions(data, output_dir)
    
    print("  - Summary Table...")
    df = create_summary_table(data, output_dir)
    
    print(f"\n✓ Analysis complete! Results saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - reward_vs_delay.png")
    print(f"  - episode_length_vs_delay.png")
    print(f"  - learning_curves.png")
    print(f"  - training_time_analysis.png")
    print(f"  - loss_convergence.png")
    print(f"  - performance_distributions.png")
    print(f"  - summary_table.png / .csv / .txt")
    
    print(f"\nSummary Statistics:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()


