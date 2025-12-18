"""
Analyze hyperparameter tuning results from the config_hp_tuning subdirectory.

This script creates comprehensive plots and tables to evaluate which configuration is best.

Usage:
    python src/analyze_hp_tuning_results.py --experiment_dir hyperparameter_tuning/tuning_20251215_232248
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_metrics_from_dir(metrics_dir):
    """Load training metrics from a metrics directory."""
    metrics_file = Path(metrics_dir) / "training_metrics.json"
    if not metrics_file.exists():
        return None
    
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        print(f"Warning: Could not load metrics from {metrics_file}: {e}")
        return None


def extract_performance_metrics(metrics):
    """Extract comprehensive performance metrics from training metrics."""
    if metrics is None:
        return None
    
    # Handle different metrics file structures
    if 'episode_metrics' in metrics:
        # New structure with episode_metrics dict
        episode_metrics = metrics.get('episode_metrics', {})
        episode_rewards = episode_metrics.get('rewards', [])
        episode_lengths = episode_metrics.get('lengths', [])
    else:
        # Old structure with direct keys
        episode_rewards = metrics.get('episode_rewards', [])
        episode_lengths = metrics.get('episode_lengths', [])
    
    if not episode_rewards:
        return None
    
    n_episodes = len(episode_rewards)
    
    # Overall statistics
    overall_mean = np.mean(episode_rewards)
    overall_std = np.std(episode_rewards)
    overall_max = np.max(episode_rewards)
    overall_min = np.min(episode_rewards)
    
    # Final 10% performance
    final_start = max(0, int(0.9 * n_episodes))
    final_rewards = episode_rewards[final_start:]
    final_mean = np.mean(final_rewards)
    final_std = np.std(final_rewards)
    final_max = np.max(final_rewards)
    
    # Final 25% performance
    final_25_start = max(0, int(0.75 * n_episodes))
    final_25_rewards = episode_rewards[final_25_start:]
    final_25_mean = np.mean(final_25_rewards)
    
    # Learning trend (compare first 25% vs last 25%)
    first_25_end = min(int(0.25 * n_episodes), n_episodes)
    first_25_rewards = episode_rewards[:first_25_end] if first_25_end > 0 else []
    first_25_mean = np.mean(first_25_rewards) if first_25_rewards else 0
    
    improvement = final_25_mean - first_25_mean
    
    # Episode length statistics
    mean_length = np.mean(episode_lengths) if episode_lengths else None
    final_length = np.mean(episode_lengths[final_start:]) if episode_lengths and len(episode_lengths) > final_start else None
    
    return {
        'total_episodes': n_episodes,
        'overall_mean_reward': overall_mean,
        'overall_std_reward': overall_std,
        'overall_max_reward': overall_max,
        'overall_min_reward': overall_min,
        'final_10pct_mean': final_mean,
        'final_10pct_std': final_std,
        'final_10pct_max': final_max,
        'final_25pct_mean': final_25_mean,
        'first_25pct_mean': first_25_mean,
        'improvement': improvement,
        'mean_episode_length': mean_length,
        'final_mean_length': final_length,
        'all_rewards': episode_rewards,  # Keep for plotting
        'all_lengths': episode_lengths if episode_lengths else [],
    }


def check_checkpoint_exists(config_dir):
    """Check if checkpoints exist for a configuration."""
    checkpoint_dir = config_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return False
    
    # Check for final checkpoint
    final_policy = checkpoint_dir / "sac_policy_final.pth"
    return final_policy.exists()


def analyze_hyperparameter_tuning(experiment_dir):
    """Analyze hyperparameter tuning results."""
    experiment_dir = Path(experiment_dir)
    
    # Handle both structures: direct config_hp_tuning or nested in experiment dir
    config_dir = experiment_dir / "config_hp_tuning"
    if not config_dir.exists():
        # Try if experiment_dir itself contains the config_hp_tuning directory
        # Or if experiment_dir is hyperparameter_tuning and config_hp_tuning is inside
        if (experiment_dir / "hyperparameter_grid.json").exists():
            # experiment_dir is actually the config_hp_tuning directory
            config_dir = experiment_dir
        else:
            print(f"Error: Could not find config_hp_tuning directory in {experiment_dir}")
            print(f"Looking for: {config_dir}")
            print(f"Or: {experiment_dir / 'hyperparameter_grid.json'}")
            return
    
    # Load hyperparameter grid
    grid_file = config_dir / "hyperparameter_grid.json"
    if not grid_file.exists():
        print(f"Error: Could not find hyperparameter_grid.json in {config_dir}")
        return
    
    with open(grid_file, 'r') as f:
        grid_data = json.load(f)
    
    configs = grid_data['configs']
    
    # Collect results for each configuration
    results = []
    
    for idx, config in enumerate(configs):
        # Find config directory (handle different naming conventions)
        config_dirs = list(config_dir.glob(f"config_{idx}_*"))
        if not config_dirs:
            continue
        
        config_dir_path = config_dirs[0]
        config_name = config_dir_path.name
        
        # Check for metrics
        metrics_dir = config_dir_path / "training_logs" / "metrics"
        metrics = load_metrics_from_dir(metrics_dir)
        performance = extract_performance_metrics(metrics)
        
        # Check for checkpoints
        has_checkpoint = check_checkpoint_exists(config_dir_path)
        
        result = {
            'config_name': config_name,
            'config_idx': idx,
            'has_metrics': metrics is not None,
            'has_checkpoint': has_checkpoint,
            **config
        }
        
        if performance:
            result.update(performance)
        else:
            # Mark as incomplete if no metrics
            result['status'] = 'incomplete' if has_checkpoint else 'missing'
        
        results.append(result)
    
    if not results:
        print("No results found.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Separate complete and incomplete configs
    df_complete = df[df['has_metrics']].copy()
    df_incomplete = df[~df['has_metrics']].copy()
    
    if len(df_complete) == 0:
        print("No configurations with complete metrics found.")
        print(f"\nConfigurations with checkpoints but no metrics: {len(df_incomplete[df_incomplete['has_checkpoint']])}")
        return
    
    # Sort by final performance
    df_complete = df_complete.sort_values('final_10pct_mean', ascending=False)
    
    # Print summary
    print("\n" + "="*100)
    print("HYPERPARAMETER TUNING RESULTS SUMMARY")
    print("="*100)
    print(f"\nTotal configurations in grid: {len(configs)}")
    print(f"Configurations with complete metrics: {len(df_complete)}")
    print(f"Configurations with checkpoints but no metrics: {len(df_incomplete[df_incomplete['has_checkpoint']])}")
    print(f"Missing configurations: {len(df_incomplete[~df_incomplete['has_checkpoint']])}")
    
    print("\n" + "-"*100)
    print("TOP CONFIGURATIONS (by final 10% mean reward):")
    print("-"*100)
    top_cols = ['config_idx', 'lr', 'hidden_dim', 'tau', 'alpha', 
                'final_10pct_mean', 'final_10pct_std', 'final_10pct_max', 
                'improvement', 'total_episodes']
    display_df = df_complete[top_cols].copy()
    display_df['final_10pct_mean'] = display_df['final_10pct_mean'].round(2)
    display_df['final_10pct_std'] = display_df['final_10pct_std'].round(2)
    display_df['final_10pct_max'] = display_df['final_10pct_max'].round(2)
    display_df['improvement'] = display_df['improvement'].round(2)
    print(display_df.to_string(index=False))
    
    print("\n" + "-"*100)
    print("BEST CONFIGURATION:")
    print("-"*100)
    best = df_complete.iloc[0]
    print(f"Config Index: {int(best['config_idx'])}")
    print(f"Config Name: {best['config_name']}")
    print(f"\nHyperparameters:")
    for key in ['lr', 'hidden_dim', 'tau', 'alpha', 'batch_size', 'update_freq']:
        if key in best:
            print(f"  {key}: {best[key]}")
    print(f"\nPerformance Metrics:")
    print(f"  Final 10% Mean Reward: {best['final_10pct_mean']:.2f} ± {best['final_10pct_std']:.2f}")
    print(f"  Final 10% Max Reward: {best['final_10pct_max']:.2f}")
    print(f"  Overall Mean Reward: {best['overall_mean_reward']:.2f} ± {best['overall_std_reward']:.2f}")
    print(f"  Overall Max Reward: {best['overall_max_reward']:.2f}")
    print(f"  Improvement (first 25% vs last 25%): {best['improvement']:.2f}")
    if best['mean_episode_length']:
        print(f"  Mean Episode Length: {best['mean_episode_length']:.1f}")
    
    # Save summary to CSV (in the same directory as config_hp_tuning)
    if config_dir == experiment_dir:
        summary_file = experiment_dir / "hyperparameter_tuning_summary.csv"
    else:
        summary_file = experiment_dir / "hyperparameter_tuning_summary.csv"
    df_complete.to_csv(summary_file, index=False)
    print(f"\nSummary saved to: {summary_file}")
    
    # Create output directory for plots (in the same directory as config_hp_tuning)
    if config_dir == experiment_dir:
        # If config_dir is the experiment_dir, create plots in parent or same dir
        plot_dir = experiment_dir / "analysis_plots"
    else:
        plot_dir = experiment_dir / "analysis_plots"
    plot_dir.mkdir(exist_ok=True)
    
    # ========== PLOT 1: Performance by Hyperparameter ==========
    hyperparams_to_plot = ['lr', 'hidden_dim', 'tau', 'alpha']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, hp in enumerate(hyperparams_to_plot):
        ax = axes[idx]
        
        # Group by hyperparameter value
        grouped = df_complete.groupby(hp)['final_10pct_mean'].agg(['mean', 'std', 'count'])
        
        x = grouped.index
        y = grouped['mean']
        yerr = grouped['std'] / np.sqrt(grouped['count'])  # Standard error
        
        bars = ax.bar(range(len(x)), y, yerr=yerr, capsize=5, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels([f"{val:.4f}" if isinstance(val, float) and val < 0.01 else str(val) for val in x], rotation=45, ha='right')
        ax.set_xlabel(hp, fontsize=12, fontweight='bold')
        ax.set_ylabel('Final 10% Mean Reward', fontsize=12, fontweight='bold')
        ax.set_title(f'Performance vs {hp}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, y)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + yerr.iloc[i] if isinstance(yerr, pd.Series) else yerr[i],
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plot_dir / "hyperparameter_analysis.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_dir / 'hyperparameter_analysis.png'}")
    
    # ========== PLOT 2: Learning Curves for Top Configurations ==========
    fig, ax = plt.subplots(figsize=(14, 8))
    
    top_n = min(5, len(df_complete))
    top_configs = df_complete.head(top_n)
    
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))
    
    for idx, (_, row) in enumerate(top_configs.iterrows()):
        config_idx = int(row['config_idx'])
        rewards = row.get('all_rewards', [])
        
        if not rewards or len(rewards) == 0:
            # Try to load from metrics file directly
            config_dirs = list((experiment_dir / "config_hp_tuning").glob(f"config_{config_idx}_*"))
            if config_dirs:
                metrics_dir = config_dirs[0] / "training_logs" / "metrics"
                metrics = load_metrics_from_dir(metrics_dir)
                if metrics:
                    if 'episode_metrics' in metrics:
                        rewards = metrics['episode_metrics'].get('rewards', [])
                    else:
                        rewards = metrics.get('episode_rewards', [])
        
        if not rewards or len(rewards) == 0:
            continue
        
        episodes = np.arange(1, len(rewards) + 1)
        
        # Compute moving average
        window = max(10, len(rewards) // 20)
        if len(rewards) > window:
            moving_avg = pd.Series(rewards).rolling(window=window, center=True).mean()
            ax.plot(episodes, moving_avg, 
                   label=f"Config {config_idx} (final: {row['final_10pct_mean']:.2f})", 
                   linewidth=2.5, alpha=0.9, color=colors[idx])
            # Also plot raw data with lower opacity
            ax.plot(episodes, rewards, alpha=0.2, color=colors[idx], linewidth=0.5)
        else:
            ax.plot(episodes, rewards, 
                   label=f"Config {config_idx} (final: {row['final_10pct_mean']:.2f})", 
                   linewidth=2.5, alpha=0.9, color=colors[idx])
    
    ax.set_xlabel('Episode', fontsize=14, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=14, fontweight='bold')
    ax.set_title(f'Learning Curves: Top {top_n} Configurations', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / "top5_learning_curves.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_dir / 'top5_learning_curves.png'}")
    
    # ========== PLOT 2B: All Learning Curves (All 15 Configs) ==========
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Get all configs from the grid (all 15)
    all_configs_list = []
    for idx, config in enumerate(configs):
        # Find the config in our results
        config_row = df[df['config_idx'] == idx]
        if len(config_row) > 0:
            all_configs_list.append(config_row.iloc[0].to_dict())
        else:
            # Create a placeholder entry for missing configs
            placeholder = {
                'config_idx': idx,
                'config_name': f'config_{idx}',
                'has_metrics': False,
                'has_checkpoint': False,
                **config
            }
            all_configs_list.append(placeholder)
    
    all_configs = pd.DataFrame(all_configs_list)
    n_configs = len(all_configs)
    
    # Use a colormap that works well for many lines
    colors = plt.cm.tab20(np.linspace(0, 1, n_configs))
    
    configs_with_data = []
    configs_without_data = []
    
    for idx, (_, row) in enumerate(all_configs.iterrows()):
        config_idx = int(row['config_idx'])
        rewards = row.get('all_rewards', [])
        
        # Check if rewards is a valid list
        if not isinstance(rewards, list) or len(rewards) == 0:
            # Try to load from metrics file directly
            config_dirs = list(config_dir.glob(f"config_{config_idx}_*"))
            if config_dirs:
                metrics_dir = config_dirs[0] / "training_logs" / "metrics"
                metrics = load_metrics_from_dir(metrics_dir)
                if metrics:
                    if 'episode_metrics' in metrics:
                        rewards = metrics['episode_metrics'].get('rewards', [])
                    else:
                        rewards = metrics.get('episode_rewards', [])
        
        if isinstance(rewards, list) and len(rewards) > 0:
            episodes = np.arange(1, len(rewards) + 1)
            
            # Compute moving average
            window = max(10, len(rewards) // 20)
            if len(rewards) > window:
                moving_avg = pd.Series(rewards).rolling(window=window, center=True).mean()
                final_mean = row.get('final_10pct_mean', np.mean(rewards[-len(rewards)//10:]) if len(rewards) > 10 else np.mean(rewards))
                ax.plot(episodes, moving_avg, 
                       label=f"Config {config_idx} (final: {final_mean:.2f})", 
                       linewidth=2, alpha=0.8, color=colors[idx])
            else:
                final_mean = np.mean(rewards) if len(rewards) > 0 else 0
                ax.plot(episodes, rewards, 
                       label=f"Config {config_idx} (final: {final_mean:.2f})", 
                       linewidth=2, alpha=0.8, color=colors[idx])
            configs_with_data.append(config_idx)
        else:
            # Config without data - show as dashed line or note
            configs_without_data.append(config_idx)
            # Add a note in the legend
            ax.plot([], [], '--', color=colors[idx], alpha=0.3, 
                   label=f"Config {config_idx} (no data)", linewidth=1)
    
    ax.set_xlabel('Episode', fontsize=14, fontweight='bold')
    ax.set_ylabel('Reward (Moving Average)', fontsize=14, fontweight='bold')
    title = f'Learning Curves: All {n_configs} Configurations'
    if configs_without_data:
        title += f' ({len(configs_with_data)} with data, {len(configs_without_data)} without)'
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=8, loc='best', ncol=3, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add text note about missing configs
    if configs_without_data:
        note_text = f"Note: Configs {', '.join(map(str, configs_without_data))} have no training metrics"
        ax.text(0.02, 0.98, note_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(plot_dir / "all_learning_curves.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_dir / 'all_learning_curves.png'}")
    if configs_without_data:
        print(f"  Note: Configs {configs_without_data} have no training metrics available")
    
    # ========== PLOT 3: Comparison Table/Heatmap ==========
    fig, ax = plt.subplots(figsize=(14, max(8, len(df_complete) * 0.4)))
    
    # Create comparison table
    table_data = []
    for _, row in df_complete.iterrows():
        table_data.append([
            int(row['config_idx']),
            f"{row['lr']:.4f}",
            row['hidden_dim'],
            f"{row['tau']:.4f}",
            row['alpha'],
            f"{row['final_10pct_mean']:.2f}",
            f"{row['final_10pct_std']:.2f}",
            f"{row['final_10pct_max']:.2f}",
            f"{row['improvement']:.2f}"
        ])
    
    columns = ['Idx', 'LR', 'Hidden', 'Tau', 'Alpha', 'Final Mean', 'Final Std', 'Final Max', 'Improvement']
    table_df = pd.DataFrame(table_data, columns=columns)
    
    # Create table
    table = ax.table(cellText=table_df.values, colLabels=table_df.columns,
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best row
    for i in range(len(columns)):
        table[(1, i)].set_facecolor('#FFD700')
        table[(1, i)].set_text_props(weight='bold')
    
    ax.axis('off')
    ax.set_title('All Configurations Comparison', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(plot_dir / "configurations_comparison_table.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_dir / 'configurations_comparison_table.png'}")
    
    # ========== PLOT 4: Performance Distribution ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Final 10% mean distribution
    ax = axes[0]
    ax.hist(df_complete['final_10pct_mean'], bins=min(10, len(df_complete)), 
           edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(df_complete['final_10pct_mean'].mean(), color='red', 
              linestyle='--', linewidth=2, label=f'Mean: {df_complete["final_10pct_mean"].mean():.2f}')
    ax.axvline(df_complete['final_10pct_mean'].max(), color='green', 
              linestyle='--', linewidth=2, label=f'Max: {df_complete["final_10pct_mean"].max():.2f}')
    ax.set_xlabel('Final 10% Mean Reward', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Configurations', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Final Performance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Improvement distribution
    ax = axes[1]
    ax.hist(df_complete['improvement'], bins=min(10, len(df_complete)), 
           edgecolor='black', alpha=0.7, color='lightcoral')
    ax.axvline(df_complete['improvement'].mean(), color='red', 
              linestyle='--', linewidth=2, label=f'Mean: {df_complete["improvement"].mean():.2f}')
    ax.set_xlabel('Improvement (Last 25% - First 25%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Configurations', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Learning Improvement', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plot_dir / "performance_distributions.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_dir / 'performance_distributions.png'}")
    
    # ========== PLOT 5: 2D Hyperparameter Interaction ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Create heatmaps for hyperparameter interactions
    interactions = [
        ('lr', 'hidden_dim', 'Learning Rate vs Hidden Dimension'),
        ('tau', 'alpha', 'Tau vs Alpha'),
        ('lr', 'tau', 'Learning Rate vs Tau'),
        ('hidden_dim', 'alpha', 'Hidden Dimension vs Alpha'),
    ]
    
    for idx, (hp1, hp2, title) in enumerate(interactions):
        ax = axes[idx]
        
        # Create pivot table
        pivot = df_complete.pivot_table(
            values='final_10pct_mean',
            index=hp1,
            columns=hp2,
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Final 10% Mean Reward'}, ax=ax, linewidths=0.5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(hp2, fontsize=11, fontweight='bold')
        ax.set_ylabel(hp1, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plot_dir / "hyperparameter_interactions.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_dir / 'hyperparameter_interactions.png'}")
    
    print(f"\n{'='*100}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*100}")
    print(f"\nAll plots saved to: {plot_dir}")
    print(f"Summary CSV saved to: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze hyperparameter tuning results')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Path to the hyperparameter tuning directory (e.g., hyperparameter_tuning)')
    
    args = parser.parse_args()
    
    analyze_hyperparameter_tuning(args.experiment_dir)

