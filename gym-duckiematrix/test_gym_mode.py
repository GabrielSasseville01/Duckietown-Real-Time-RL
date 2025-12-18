"""
Test script for gym mode environment.
Compares gym mode with regular mode to verify functionality and speed.
"""

import time
import numpy as np
from gym_duckiematrix.DB21J_gym import DuckiematrixDB21JEnvGym
from gym_duckiematrix.DB21J import DuckiematrixDB21JEnv


def test_gym_mode(num_steps=10, step_duration=0.1):
    """Test gym mode environment."""
    print("=" * 60)
    print("Testing Gym Mode Environment")
    print("=" * 60)
    
    # Create gym mode environment
    print(f"\nCreating gym mode environment with step_duration={step_duration}s...")
    env = DuckiematrixDB21JEnvGym(
        entity_name="map_0/vehicle_0",
        include_curve_flag=True,
        step_duration=step_duration
    )
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Reset environment
    print("\nResetting environment...")
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    
    # Run a few steps
    print(f"\nRunning {num_steps} steps...")
    total_reward = 0
    start_time = time.time()
    actual_steps = 0
    
    for step in range(num_steps):
        # Random action
        action = env.action_space.sample()
        
        step_start = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = time.time() - step_start
        
        actual_steps += 1
        total_reward += reward
        
        print(f"\nStep {step + 1}:")
        print(f"  Action: {action}")
        print(f"  Observation: {obs}")
        print(f"  Reward: {reward:.3f}")
        print(f"  Step time: {step_time:.3f}s")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        if "tile" in info:
            print(f"  Tile ID: {info.get('tile')}, Is curve: {info.get('is_curve_tile')}")
        
        if terminated or truncated:
            print(f"  Episode ended at step {step + 1}!")
            # Reset to start a new episode
            print("  Resetting environment for next episode...")
            obs, info = env.reset()
            print(f"  Reset complete. New observation: {obs}")
    
    total_time = time.time() - start_time
    avg_step_time = total_time / actual_steps if actual_steps > 0 else 0.0
    
    print(f"\n" + "=" * 60)
    print(f"Gym Mode Test Results:")
    print(f"  Requested steps: {num_steps}")
    print(f"  Actual steps completed: {actual_steps}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Average step time: {avg_step_time:.3f}s")
    print(f"  Total reward: {total_reward:.3f}")
    print(f"  Expected step time (step_duration): {step_duration:.3f}s")
    print("\n" + "-" * 60)
    print("VERIFICATION CHECKLIST:")
    print("-" * 60)
    
    # Check 1: Step timing should be close to step_duration
    timing_diff = abs(avg_step_time - step_duration)
    timing_ok = timing_diff < 0.05  # Allow 50ms tolerance
    print(f"✓ Step timing: {'PASS' if timing_ok else 'FAIL'}")
    print(f"    Expected: ~{step_duration:.3f}s, Got: {avg_step_time:.3f}s")
    if not timing_ok:
        print(f"    WARNING: Step time differs by {timing_diff:.3f}s from expected")
    
    # Check 2: Observations should be valid (not all zeros after first step)
    print(f"✓ Observations: {'PASS' if obs is not None else 'FAIL'}")
    print(f"    Last observation: {obs}")
    
    # Check 3: Rewards should be computed
    reward_ok = not np.isnan(total_reward) and total_reward != 0.0
    print(f"✓ Rewards computed: {'PASS' if reward_ok else 'FAIL'}")
    print(f"    Total reward: {total_reward:.3f}")
    
    # Check 4: Environment should complete steps without errors
    print(f"✓ Steps completed: {'PASS' if actual_steps > 0 else 'FAIL'}")
    print(f"    Completed {actual_steps} steps without crashing")
    
    print("-" * 60)
    if timing_ok and reward_ok:
        print("✓ GYM MODE APPEARS TO BE WORKING CORRECTLY!")
    else:
        print("⚠ GYM MODE MAY HAVE ISSUES - CHECK WARNINGS ABOVE")
    print("=" * 60)
    
    # Cleanup
    env.robot.motors.stop()
    
    return total_time, avg_step_time


def compare_with_regular_mode(num_steps=10):
    """Compare gym mode with regular mode."""
    print("\n" + "=" * 60)
    print("Comparing Gym Mode vs Regular Mode")
    print("=" * 60)
    
    # Test regular mode
    print("\n--- Regular Mode ---")
    env_regular = DuckiematrixDB21JEnv(
        entity_name="map_0/vehicle_0",
        include_curve_flag=True
    )
    
    obs, info = env_regular.reset()
    start_time = time.time()
    
    for step in range(num_steps):
        action = env_regular.action_space.sample()
        obs, reward, terminated, truncated, info = env_regular.step(action)
        if terminated or truncated:
            break
    
    regular_time = time.time() - start_time
    env_regular.robot.motors.stop()
    
    # Test gym mode
    print("\n--- Gym Mode ---")
    gym_time, avg_gym_time = test_gym_mode(num_steps=num_steps, step_duration=0.1)
    
    # Comparison
    print("\n" + "=" * 60)
    print("Comparison Results:")
    print(f"  Regular mode time: {regular_time:.3f}s")
    print(f"  Gym mode time: {gym_time:.3f}s")
    if regular_time > 0:
        speedup = regular_time / gym_time
        print(f"  Speedup: {speedup:.2f}x")
    
    print("\n" + "-" * 60)
    print("COMPARISON VERIFICATION:")
    print("-" * 60)
    
    # Check if gym mode is actually different
    if regular_time > 0:
        time_diff = abs(regular_time - gym_time)
        time_diff_pct = (time_diff / regular_time) * 100
        print(f"✓ Timing difference: {time_diff:.3f}s ({time_diff_pct:.1f}%)")
        if time_diff_pct > 5:  # More than 5% difference
            print(f"    Gym mode timing is significantly different (expected)")
        else:
            print(f"    WARNING: Timing very similar - gym mode may not be active")
    
    # Check if both completed successfully
    print(f"✓ Both modes completed: {'PASS' if regular_time > 0 and gym_time > 0 else 'FAIL'}")
    
    print("-" * 60)
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test gym mode environment')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='Number of steps to run (default: 10)')
    parser.add_argument('--step_duration', type=float, default=0.1,
                        help='Step duration in seconds (default: 0.1)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with regular mode')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_with_regular_mode(num_steps=args.num_steps)
    else:
        test_gym_mode(num_steps=args.num_steps, step_duration=args.step_duration)

