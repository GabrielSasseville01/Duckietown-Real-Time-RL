"""
Test the time it takes to sample actions from SAC policy network.
Samples 10,000 actions and plots the distribution of sampling times.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from sac_agent import SACAgent

def test_sac_sampling_time(num_samples=10000):
    """
    Test action sampling time for SAC agent.
    
    Args:
        num_samples: Number of actions to sample
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create SAC agent
    obs_dim = 3  # Assuming include_curve_flag=True
    action_dim = 2
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        device=device
    )
    
    # Create a dummy observation (typical values)
    obs = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # [d_signed, theta, curve_flag]
    
    print(f"\nSampling {num_samples:,} actions from SAC policy network...")
    print("This may take a moment...\n")
    
    # Time each sample
    sampling_times = []
    
    # Warmup (first few samples might be slower)
    for _ in range(10):
        _ = agent.select_action(obs, deterministic=False, apply_exploration=True)
    
    # Actual timing
    start_total = time.perf_counter()
    
    for i in range(num_samples):
        start = time.perf_counter()
        action = agent.select_action(obs, deterministic=False, apply_exploration=True)
        end = time.perf_counter()
        
        sampling_time = (end - start) * 1000  # Convert to milliseconds
        sampling_times.append(sampling_time)
        
        if (i + 1) % 1000 == 0:
            print(f"  Sampled {i+1:,}/{num_samples:,} actions...")
    
    end_total = time.perf_counter()
    total_time = end_total - start_total
    
    # Convert to numpy array for analysis
    sampling_times = np.array(sampling_times)
    
    # Statistics
    print(f"\n{'='*60}")
    print("SAMPLING TIME STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples: {num_samples:,}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per sample: {np.mean(sampling_times):.4f} ms")
    print(f"Median time per sample: {np.median(sampling_times):.4f} ms")
    print(f"Min time: {np.min(sampling_times):.4f} ms")
    print(f"Max time: {np.max(sampling_times):.4f} ms")
    print(f"Std deviation: {np.std(sampling_times):.4f} ms")
    print(f"95th percentile: {np.percentile(sampling_times, 95):.4f} ms")
    print(f"99th percentile: {np.percentile(sampling_times, 99):.4f} ms")
    print(f"\nSamples per second: {num_samples / total_time:.0f}")
    
    # Set style for better presentation plots
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.titlesize'] = 20
    
    # Create histogram (single figure)
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    ax.hist(sampling_times, bins=100, edgecolor='black', alpha=0.7, color='#2ca02c')
    ax.axvline(np.mean(sampling_times), color='red', linestyle='--', linewidth=3, label=f'Mean: {np.mean(sampling_times):.4f} ms')
    ax.axvline(np.median(sampling_times), color='blue', linestyle='--', linewidth=3, label=f'Median: {np.median(sampling_times):.4f} ms')
    ax.set_xlabel('Sampling Time (ms)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=18, fontweight='bold')
    ax.set_title(f'SAC Action Sampling Time Distribution ({num_samples:,} samples)', fontsize=20, fontweight='bold', pad=20)
    ax.legend(fontsize=16, framealpha=0.9, edgecolor='black', frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.tick_params(axis='both', which='major', labelsize=16, width=1.5, length=6)
    
    plt.tight_layout(pad=2.0)
    
    # Save plot
    plot_file = 'sac_sampling_time_distribution.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Plot saved to {plot_file}")
    
    plot_file_pdf = 'sac_sampling_time_distribution.pdf'
    plt.savefig(plot_file_pdf, bbox_inches='tight', facecolor='white')
    print(f"✓ Plot saved to {plot_file_pdf}")
    
    plt.close()
    
    # Create presentation figure with shifted distribution
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 10))
    
    # Shift amount (e.g., add 2ms to simulate slower network)
    shift_amount = 2.0  # ms
    
    # Original distribution (faded)
    ax2.hist(sampling_times, bins=100, edgecolor='black', alpha=0.3, color='#2ca02c', 
            label='Fast Network (Original)')
    
    # Shifted distribution (red)
    shifted_times = sampling_times + shift_amount
    ax2.hist(shifted_times, bins=100, edgecolor='darkred', alpha=0.7, color='red', 
            label=f'Slow Network (Shifted +{shift_amount} ms)')
    
    # Add mean lines
    ax2.axvline(np.mean(sampling_times), color='green', linestyle='--', linewidth=3, 
               alpha=0.6, label=f'Fast Mean: {np.mean(sampling_times):.4f} ms')
    ax2.axvline(np.mean(shifted_times), color='darkred', linestyle='--', linewidth=3, 
               label=f'Slow Mean: {np.mean(shifted_times):.4f} ms')
    
    ax2.set_xlabel('Sampling Time (ms)', fontsize=18, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=18, fontweight='bold')
    ax2.set_title('SAC Action Sampling Time: Fast vs Slow Network Comparison', 
                 fontsize=20, fontweight='bold', pad=20)
    ax2.legend(fontsize=16, loc='upper right', framealpha=0.9, edgecolor='black', frameon=True)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.tick_params(axis='both', which='major', labelsize=16, width=1.5, length=6)
    
    plt.tight_layout(pad=2.0)
    
    # Save presentation plot
    presentation_file = 'sac_sampling_time_presentation.png'
    plt.savefig(presentation_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Presentation plot saved to {presentation_file}")
    
    presentation_file_pdf = 'sac_sampling_time_presentation.pdf'
    plt.savefig(presentation_file_pdf, bbox_inches='tight', facecolor='white')
    print(f"✓ Presentation plot saved to {presentation_file_pdf}")
    
    plt.close()
    
    # Also create a box plot
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    bp = ax.boxplot([sampling_times], tick_labels=['SAC Policy'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ca02c')
    bp['boxes'][0].set_alpha(0.7)
    ax.set_ylabel('Sampling Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title(f'SAC Action Sampling Time - Box Plot ({num_samples:,} samples)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    boxplot_file = 'sac_sampling_time_boxplot.png'
    plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Box plot saved to {boxplot_file}")
    
    plt.close()
    
    print(f"\n{'='*60}")
    print("Test complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test SAC action sampling time')
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of actions to sample (default: 10000)')
    
    args = parser.parse_args()
    
    test_sac_sampling_time(num_samples=args.num_samples)

