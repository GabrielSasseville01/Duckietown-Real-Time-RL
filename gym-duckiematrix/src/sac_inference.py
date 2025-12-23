"""
Run inference/evaluation with a trained SAC agent.
No exploration - uses deterministic policy.
"""

import torch
import numpy as np
import argparse
import time
from gym_duckiematrix.DB21J import DuckiematrixDB21JEnv
from gym_duckiematrix.DB21J_gym import DuckiematrixDB21JEnvGym
from sac_agent import SACAgent
from time import sleep
from typing import Optional


def _set_global_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_inference(policy_path, q1_path=None, q2_path=None, num_episodes=10, max_steps=2000,
                  render=True, use_gym_mode=False, step_duration=0.1,
                  condition_on_prev_action: bool = False,
                  # Experiment 3: variable delay config (used only in gym_mode)
                  delay_mode: str = "fixed",
                  delay_dist: str = "lognormal",
                  delay_mean: Optional[float] = None,
                  delay_std: Optional[float] = None,
                  delay_cv: Optional[float] = None,
                  delay_min: float = 0.0,
                  delay_max: Optional[float] = None,
                  delay_seed: Optional[int] = None,
                  delay_spike_prob: float = 0.1,
                  delay_spike_multiplier: float = 4.0,
                  seed: Optional[int] = None):
    """
    Run inference with a trained SAC agent.
    
    Args:
        policy_path: Path to saved policy network
        q1_path: Path to saved Q1 network (optional)
        q2_path: Path to saved Q2 network (optional)
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        render: Whether to add small delay for visualization
        use_gym_mode: Whether to use gym mode (faster, non-real-time) (default: False)
        step_duration: Step duration for gym mode in seconds (default: 0.1)
        condition_on_prev_action: Whether to include previous action in observations (default: False)
    """
    if seed is not None:
        print(f"Setting global seed: {seed}")
        _set_global_seeds(int(seed))
        if delay_seed is None:
            delay_seed = int(seed)

    # Create environment (gym mode or regular mode)
    if use_gym_mode:
        print(f"Using GYM MODE (step_duration={step_duration}s)")
        if condition_on_prev_action:
            print(f"Condition on Previous Action: ENABLED")
        env = DuckiematrixDB21JEnvGym(
            entity_name="map_0/vehicle_0", 
            include_curve_flag=True,
            step_duration=step_duration,
            condition_on_prev_action=condition_on_prev_action,
            delay_mode=delay_mode,
            delay_dist=delay_dist,
            delay_mean=delay_mean,
            delay_std=delay_std,
            delay_cv=delay_cv,
            delay_min=delay_min,
            delay_max=delay_max,
            delay_seed=delay_seed,
            delay_spike_prob=delay_spike_prob,
            delay_spike_multiplier=delay_spike_multiplier,
        )
    else:
        print("Using REGULAR MODE (real-time)")
        if condition_on_prev_action:
            print(f"Condition on Previous Action: ENABLED")
        env = DuckiematrixDB21JEnv(entity_name="map_0/vehicle_0", include_curve_flag=True)
    
    # Create agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=3e-4,  # Not used during inference
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        device=device,
    )
    
    # Load checkpoint
    agent.load_checkpoint(policy_path, q1_path, q2_path)
    print(f"Loaded model from {policy_path}")
    print("Running in deterministic mode (no exploration)\n")

    # Quick sanity-check (helps catch mismatched action-conditioning during eval)
    try:
        loaded_policy_obs_dim = agent.policy.fc1.in_features
        if loaded_policy_obs_dim != obs_dim:
            print("WARNING: Observation dimension mismatch!")
            print(f"  Checkpoint policy expects obs_dim={loaded_policy_obs_dim}")
            print(f"  Environment provides obs_dim={obs_dim}")
            print("  Make sure --condition_on_prev_action matches training.")
    except Exception:
        pass
    
    # Benchmark forward pass time
    print("Benchmarking forward pass time...")
    forward_times = []
    num_warmup = 10
    num_tests = 10000
    
    # Warmup with varying observations
    for _ in range(num_warmup):
        test_obs = env.observation_space.sample()
        _ = agent.select_action(test_obs, deterministic=True, apply_exploration=False)
    
    # Measure forward pass times with varying observations (more realistic)
    # If using GPU, synchronize to get accurate timing
    if device == 'cuda':
        torch.cuda.synchronize()
    
    for _ in range(num_tests):
        test_obs = env.observation_space.sample()  # Different observation each time
        if device == 'cuda':
            torch.cuda.synchronize()  # Wait for any pending GPU operations
        start_time = time.perf_counter()
        _ = agent.select_action(test_obs, deterministic=True, apply_exploration=False)
        if device == 'cuda':
            torch.cuda.synchronize()  # Wait for GPU operations to complete
        end_time = time.perf_counter()
        forward_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    avg_time = np.median(forward_times)
    std_time = np.std(forward_times)
    min_time = np.min(forward_times)
    max_time = np.max(forward_times)
    
    print(f"Forward pass timing (over {num_tests} runs):")
    print(f"  Average: {avg_time:.3f} ms")
    print(f"  Std Dev: {std_time:.3f} ms")
    print(f"  Min: {min_time:.3f} ms")
    print(f"  Max: {max_time:.3f} ms")
    print(f"  Frequency: {1000/avg_time:.1f} Hz (actions per second)\n")
    
    # Statistics
    episode_rewards = []
    episode_lengths = []
    action_times = []  # Track action selection times during episodes
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        print(f"Episode {episode + 1}/{num_episodes}")
        
        for step in range(max_steps):
            # Select action deterministically (no exploration) and measure time
            if device == 'cuda':
                torch.cuda.synchronize()  # Wait for any pending GPU operations
            action_start = time.perf_counter()
            action = agent.select_action(obs, deterministic=True, apply_exploration=False)
            if device == 'cuda':
                torch.cuda.synchronize()  # Wait for GPU operations to complete
            action_end = time.perf_counter()
            action_time = (action_end - action_start) * 1000  # Convert to milliseconds
            action_times.append(action_time)
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Only add delay in regular mode (gym mode handles timing internally)
            if render and not use_gym_mode:
                sleep(0.01)  # Small delay for visualization
            
            # Check if episode is done
            if terminated or truncated:
                print(f"  Terminated at step {step + 1}")
                break
            
            obs = next_obs
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"  Reward: {episode_reward:.2f}, Length: {episode_length}\n")
    
    # Calculate summary statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    max_reward = np.max(episode_rewards)
    min_reward = np.min(episode_rewards)
    
    # Print summary
    print("=" * 50)
    print("Inference Summary:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Avg Length: {avg_length:.1f} ± {std_length:.1f}")
    print(f"  Max Reward: {max_reward:.2f}")
    print(f"  Min Reward: {min_reward:.2f}")
    print()
    print("Action Selection Timing (during episodes):")
    if action_times:
        print(f"  Average: {np.median(action_times):.3f} ms")
        print(f"  Std Dev: {np.std(action_times):.3f} ms")
        print(f"  Min: {np.min(action_times):.3f} ms")
        print(f"  Max: {np.max(action_times):.3f} ms")
        print(f"  Total actions: {len(action_times)}")
    print("=" * 50)
    
    # Cleanup
    env.robot.motors.stop()
    
    # Return metrics dictionary for easier parsing
    metrics = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "avg_reward": float(avg_reward),
        "reward_std": float(std_reward),
        "avg_length": float(avg_length),
        "length_std": float(std_length),
        "max_reward": float(max_reward),
        "min_reward": float(min_reward),
        "num_episodes": num_episodes,
    }
    
    return episode_rewards, episode_lengths, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with trained SAC agent')
    parser.add_argument('--policy_checkpoint', type=str, required=True,
                        help='Path to policy checkpoint (required)')
    parser.add_argument('--q1_checkpoint', type=str, default=None,
                        help='Path to Q1 checkpoint (optional)')
    parser.add_argument('--q2_checkpoint', type=str, default=None,
                        help='Path to Q2 checkpoint (optional)')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of episodes to run (default: 10)')
    parser.add_argument('--max_steps', type=int, default=2000,
                        help='Maximum steps per episode (default: 2000)')
    parser.add_argument('--no_render', action='store_true',
                        help='Disable rendering delay (faster)')
    parser.add_argument('--gym_mode', action='store_true',
                        help='Use gym mode (faster, non-real-time simulation)')
    parser.add_argument('--step_duration', type=float, default=0.1,
                        help='Step duration for gym mode in seconds (default: 0.1)')
    parser.add_argument('--save_metrics', action='store_true',
                        help='Save evaluation metrics to JSON file')
    parser.add_argument('--condition_on_prev_action', action='store_true',
                        help='Include previous action in observations (MUST match training if enabled)')
    parser.add_argument('--action-conditioning', dest='condition_on_prev_action', action='store_true',
                        help='Alias for --condition_on_prev_action')

    # Experiment 3: variable delay distribution (used only when --gym_mode)
    parser.add_argument('--delay_mode', type=str, default='fixed', choices=['fixed', 'random'],
                        help='Delay mode: fixed uses --step_duration; random samples per step (default: fixed)')
    parser.add_argument('--delay_dist', type=str, default='lognormal',
                        choices=['uniform', 'normal', 'lognormal', 'exponential', 'mixture'],
                        help='Distribution for random delays (default: lognormal)')
    parser.add_argument('--delay_mean', type=float, default=None,
                        help='Mean delay (seconds) for random mode (default: uses --step_duration)')
    parser.add_argument('--delay_std', type=float, default=None,
                        help='Std dev (seconds) for random mode (overrides --delay_cv)')
    parser.add_argument('--delay_cv', type=float, default=None,
                        help='Coefficient of variation std/mean for random mode (e.g., 0.2)')
    parser.add_argument('--delay_min', type=float, default=0.0,
                        help='Minimum delay bound (seconds) (default: 0.0)')
    parser.add_argument('--delay_max', type=float, default=None,
                        help='Maximum delay bound (seconds) (default: unbounded)')
    parser.add_argument('--delay_seed', type=int, default=None,
                        help='Seed for delay RNG (default: uses --seed if provided)')
    parser.add_argument('--delay_spike_prob', type=float, default=0.1,
                        help='For delay_dist=mixture: probability of spikes (default: 0.1)')
    parser.add_argument('--delay_spike_multiplier', type=float, default=4.0,
                        help='For delay_dist=mixture: spike mean = delay_mean * multiplier (default: 4.0)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Global seed for numpy/torch/random (and delay RNG if delay_seed not set)')
    
    args = parser.parse_args()
    
    # Run inference
    rewards, lengths, metrics = run_inference(
        policy_path=args.policy_checkpoint,
        q1_path=args.q1_checkpoint,
        q2_path=args.q2_checkpoint,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        render=not args.no_render,
        use_gym_mode=args.gym_mode,
        step_duration=args.step_duration,
        condition_on_prev_action=args.condition_on_prev_action,
        delay_mode=args.delay_mode,
        delay_dist=args.delay_dist,
        delay_mean=args.delay_mean,
        delay_std=args.delay_std,
        delay_cv=args.delay_cv,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        delay_seed=args.delay_seed,
        delay_spike_prob=args.delay_spike_prob,
        delay_spike_multiplier=args.delay_spike_multiplier,
        seed=args.seed,
    )
    
    # Optionally save metrics to file
    if args.save_metrics:
        import json
        from pathlib import Path
        
        metrics_file = Path(args.policy_checkpoint).parent / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nEvaluation metrics saved to: {metrics_file}")

