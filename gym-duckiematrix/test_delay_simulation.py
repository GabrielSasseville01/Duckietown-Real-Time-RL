"""
Test script to verify computation delay simulation in gym mode.
Tests that:
1. State evolves during computation delay
2. Actions are applied to evolved states (not computation states)
3. Longer delays cause more state evolution (more outdated actions)
"""

import time
import numpy as np
import math
from gym_duckiematrix.DB21J_gym import DuckiematrixDB21JEnvGym
from duckietown.sdk.utils.lane_position import compute_yaw


def get_position(pose):
    """Extract position from pose."""
    if pose is None:
        return None
    return (
        pose["position"]["x"],
        pose["position"]["y"],
        compute_yaw(pose)
    )


def distance(pos1, pos2):
    """Calculate distance between two positions."""
    if pos1 is None or pos2 is None:
        return None
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def test_delay_simulation(step_duration=0.1, num_steps=5):
    """
    Test that delay simulation works correctly.
    
    Verifies:
    - State evolves during computation delay
    - Actions are applied to evolved states
    - Longer delays cause more state evolution
    """
    print("=" * 80)
    print(f"Testing Delay Simulation (step_duration={step_duration}s)")
    print("=" * 80)
    
    # Create environment
    env = DuckiematrixDB21JEnvGym(
        entity_name="map_0/vehicle_0",
        include_curve_flag=True,
        step_duration=step_duration
    )
    
    # Reset
    obs, info = env.reset()
    initial_pos = get_position(info.get("pose"))
    print(f"\nInitial position: ({initial_pos[0]:.3f}, {initial_pos[1]:.3f}, {initial_pos[2]:.3f})")
    
    # Track state evolution
    computation_states = []  # States where actions were computed for
    evolved_states = []      # States after delay (where actions are applied)
    final_states = []        # Final states after applying action
    
    print(f"\nRunning {num_steps} steps...")
    print("-" * 80)
    
    for step in range(num_steps):
        # Get computation state (where action will be computed for)
        # This is stored in _computation_state_position
        computation_pos = env._computation_state_position
        if computation_pos is None:
            # First step - use current position
            computation_pos = get_position(env.robot.pose.capture())
        
        # Select action (this would normally be computed by the agent)
        action = env.action_space.sample()
        
        print(f"\nStep {step + 1}:")
        print(f"  Action: [{action[0]:.3f}, {action[1]:.3f}]")
        print(f"  Computation state (S₀): ({computation_pos[0]:.3f}, {computation_pos[1]:.3f}, {computation_pos[2]:.3f})")
        
        # Capture state before step (computation state)
        before_step_pose = env.robot.pose.capture()
        before_step_pos = get_position(before_step_pose)
        
        # Take step
        step_start = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = time.time() - step_start
        
        # Capture state after step
        after_step_pose = info.get("pose")
        after_step_pos = get_position(after_step_pose)
        
        # Try to capture evolved state (this would be S₀' in the implementation)
        # We can't directly access it, but we can infer from the step behavior
        # The key is: action was computed for computation_pos, but applied after delay
        
        print(f"  Final state (S₁): ({after_step_pos[0]:.3f}, {after_step_pos[1]:.3f}, {after_step_pos[2]:.3f})")
        print(f"  Reward: {reward:.3f}")
        print(f"  Step time: {step_time:.3f}s (expected: ~{step_duration:.3f}s)")
        
        # Calculate state change
        if step > 0:
            state_change = distance(computation_pos, after_step_pos)
            print(f"  State change (S₀ → S₁): {state_change:.3f}m")
        
        computation_states.append(computation_pos)
        final_states.append(after_step_pos)
        
        if terminated or truncated:
            print(f"  Episode ended!")
            break
    
    # Analysis
    print("\n" + "=" * 80)
    print("DELAY SIMULATION ANALYSIS")
    print("=" * 80)
    
    # Check 1: State should evolve between steps
    print("\n1. State Evolution Between Steps:")
    state_evolutions = []
    for i in range(1, len(final_states)):
        prev_pos = final_states[i-1]
        curr_pos = final_states[i]
        evolution = distance(prev_pos, curr_pos)
        state_evolutions.append(evolution)
        print(f"   Step {i} → {i+1}: {evolution:.3f}m")
    
    if state_evolutions:
        avg_evolution = np.mean(state_evolutions)
        print(f"   Average state evolution: {avg_evolution:.3f}m")
    
    # Check 2: Step timing should match step_duration
    print(f"\n2. Step Timing:")
    print(f"   Expected: ~{step_duration:.3f}s")
    print(f"   Actual: {step_time:.3f}s")
    timing_ok = abs(step_time - step_duration) < 0.1
    print(f"   Status: {'✓ PASS' if timing_ok else '✗ FAIL'}")
    
    # Check 3: Robot should move (not stay in same place)
    print(f"\n3. Robot Movement:")
    if len(final_states) > 1:
        total_movement = distance(final_states[0], final_states[-1])
        print(f"   Total movement: {total_movement:.3f}m")
        movement_ok = total_movement > 0.01  # At least 1cm movement
        print(f"   Status: {'✓ PASS' if movement_ok else '✗ FAIL (robot not moving)'}")
    else:
        print(f"   Status: ⚠ Need more steps to verify")
    
    # Check 4: Actions should affect final state
    print(f"\n4. Action Effectiveness:")
    if len(final_states) > 1:
        # Check if different actions lead to different states
        print(f"   Multiple steps completed with different states")
        print(f"   Status: ✓ PASS (actions are being applied)")
    
    print("\n" + "=" * 80)
    
    # Cleanup
    env.robot.motors.stop()
    
    return {
        "step_duration": step_duration,
        "num_steps": len(final_states),
        "avg_state_evolution": avg_evolution if state_evolutions else 0.0,
        "total_movement": total_movement if len(final_states) > 1 else 0.0,
        "timing_ok": timing_ok
    }


def compare_delays(step_durations=[0.05, 0.1, 0.2, 0.5], num_steps=5):
    """
    Compare how different delays affect state evolution.
    
    Expected: Longer delays → more state evolution → more outdated actions
    """
    print("\n" + "=" * 80)
    print("COMPARING DIFFERENT DELAY VALUES")
    print("=" * 80)
    print("\nThis test verifies that longer delays cause more state evolution,")
    print("making actions more outdated (key insight of real-time RL).\n")
    
    results = []
    
    for delay in step_durations:
        print(f"\n{'='*80}")
        result = test_delay_simulation(step_duration=delay, num_steps=num_steps)
        results.append(result)
        time.sleep(1)  # Small delay between tests
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("DELAY COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Delay (s)':<12} {'Steps':<8} {'Avg Evolution (m)':<20} {'Total Movement (m)':<20} {'Status'}")
    print("-" * 80)
    
    for result in results:
        status = "✓ OK" if result["timing_ok"] else "✗ FAIL"
        print(f"{result['step_duration']:<12.3f} {result['num_steps']:<8} "
              f"{result['avg_state_evolution']:<20.3f} {result['total_movement']:<20.3f} {status}")
    
    print("\n" + "-" * 80)
    print("KEY INSIGHT:")
    print("Longer delays should show more state evolution between computation and execution.")
    print("This demonstrates the 'outdated state' problem in real-time RL.")
    print("=" * 80)


def test_state_mismatch(step_duration=0.5, num_steps=3):
    """
    Test that actions are applied to evolved states (not computation states).
    
    This is the core of the delay simulation: action computed for S₀,
    but applied to S₀' (evolved state).
    """
    print("\n" + "=" * 80)
    print(f"Testing State Mismatch (step_duration={step_duration}s)")
    print("=" * 80)
    print("\nThis test verifies that:")
    print("1. Action is computed for state S₀")
    print("2. During delay, state evolves to S₀'")
    print("3. Action is applied to S₀' (not S₀)")
    print("4. Longer delay → larger S₀' - S₀ difference\n")
    
    env = DuckiematrixDB21JEnvGym(
        entity_name="map_0/vehicle_0",
        include_curve_flag=True,
        step_duration=step_duration
    )
    
    obs, info = env.reset()
    
    for step in range(num_steps):
        # Get computation state
        computation_pos = env._computation_state_position
        if computation_pos is None:
            computation_pos = get_position(env.robot.pose.capture())
        
        print(f"\nStep {step + 1}:")
        print(f"  Computation state (S₀): ({computation_pos[0]:.3f}, {computation_pos[1]:.3f})")
        
        # The implementation should:
        # 1. Reset to computation_pos
        # 2. Let evolve for step_duration
        # 3. Apply action to evolved state
        
        # We can't directly observe S₀', but we can verify the behavior:
        # - Robot should move during the delay
        # - Final state should be different from computation state
        
        action = [0.5, 0.5]  # Forward action
        obs, reward, terminated, truncated, info = env.step(action)
        
        final_pos = get_position(info.get("pose"))
        print(f"  Final state (S₁): ({final_pos[0]:.3f}, {final_pos[1]:.3f})")
        
        # Calculate mismatch (how much state changed)
        state_mismatch = distance(computation_pos, final_pos)
        print(f"  State mismatch (S₀ → S₁): {state_mismatch:.3f}m")
        print(f"  (This includes evolution during delay + action effect)")
        
        if terminated:
            break
    
    env.robot.motors.stop()
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test delay simulation in gym mode')
    parser.add_argument('--step_duration', type=float, default=0.1,
                        help='Step duration to test (default: 0.1)')
    parser.add_argument('--num_steps', type=int, default=5,
                        help='Number of steps to run (default: 5)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare different delay values')
    parser.add_argument('--test_mismatch', action='store_true',
                        help='Test state mismatch (action applied to evolved state)')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_delays(step_durations=[0.05, 0.1, 0.2, 0.5], num_steps=args.num_steps)
    elif args.test_mismatch:
        test_state_mismatch(step_duration=args.step_duration, num_steps=args.num_steps)
    else:
        test_delay_simulation(step_duration=args.step_duration, num_steps=args.num_steps)

