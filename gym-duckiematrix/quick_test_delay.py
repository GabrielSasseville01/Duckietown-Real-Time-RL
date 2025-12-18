"""
Quick test to verify delay simulation is working.
Run this first to make sure everything is set up correctly.
"""

import time
import numpy as np
from gym_duckiematrix.DB21J_gym import DuckiematrixDB21JEnvGym

def quick_test():
    """Quick test of delay simulation."""
    print("=" * 60)
    print("Quick Delay Simulation Test")
    print("=" * 60)
    
    # Test with different delays
    delays = [3.0]
    
    for delay in delays:
        print(f"\n{'='*60}")
        print(f"Testing with step_duration = {delay}s")
        print(f"{'='*60}\n")
        
        env = DuckiematrixDB21JEnvGym(
            entity_name="map_0/vehicle_0",
            include_curve_flag=True,
            step_duration=delay
        )
        
        obs, info = env.reset()
        initial_pos = info["pose"]["position"]
        print(f"Initial position: ({initial_pos['x']:.3f}, {initial_pos['y']:.3f})")
        
        total_reward = 0
        positions = []
        
        for step in range(5):
            action = [0.5, 0.5]  # Forward action
            
            # Capture position before step
            before_pose = env.robot.pose.capture()
            before_pos = (before_pose["position"]["x"], before_pose["position"]["y"]) if before_pose else None
            
            print(f"\nStep {step + 1}:")
            if before_pos:
                print(f"  Position BEFORE step: ({before_pos[0]:.3f}, {before_pos[1]:.3f})")
            print(f"  Action to apply: [{action[0]:.3f}, {action[1]:.3f}]")
            print(f"  Expected: Robot should move forward for {delay}s during this step")
            
            step_start = time.time()
            print(f"  Starting step (this will take ~{delay}s)...")
            obs, reward, terminated, truncated, info = env.step(action)
            step_time = time.time() - step_start
            print(f"  Step completed in {step_time:.3f}s")
            
            pos = info["pose"]["position"]
            positions.append((pos['x'], pos['y']))
            
            print(f"  Position AFTER step: ({pos['x']:.3f}, {pos['y']:.3f})")
            if before_pos:
                movement = ((pos['x'] - before_pos[0])**2 + (pos['y'] - before_pos[1])**2)**0.5
                print(f"  Movement this step: {movement:.3f}m")
            print(f"  Reward: {reward:.3f}")
            print(f"  Step time: {step_time:.3f}s (expected: ~{delay:.3f}s)")
            
            total_reward += reward
            
            if terminated or truncated:
                print(f"  Episode ended!")
                break
        
        # Check movement
        if len(positions) > 1:
            import math
            movement = math.sqrt(
                (positions[-1][0] - positions[0][0])**2 + 
                (positions[-1][1] - positions[0][1])**2
            )
            print(f"\nTotal movement: {movement:.3f}m")
            print(f"Total reward: {total_reward:.3f}")
        
        # Verify timing
        timing_ok = abs(step_time - delay) < 0.1
        print(f"\nTiming check: {'✓ PASS' if timing_ok else '✗ FAIL'}")
        
        env.robot.motors.stop()
        time.sleep(0.5)  # Small delay between tests
    
    print("\n" + "=" * 60)
    print("Quick test complete!")
    print("=" * 60)
    print("\nIf timing checks pass and robot moves, delay simulation is working!")
    print("Run 'python test_delay_simulation.py --compare' for detailed analysis.")

if __name__ == "__main__":
    quick_test()

