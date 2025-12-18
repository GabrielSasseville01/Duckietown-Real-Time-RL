"""
Simple script to check CURVED_TILES definition.
You can run this during training to see which tiles are marked as curved.
Or check the info dict in your training loop.
"""
from src.gym_duckiematrix.DB21J import CURVED_TILES

print("Current CURVED_TILES definition:", CURVED_TILES)
print("\nThis means:")
print(f"  Curved tiles: {sorted(CURVED_TILES)}")
print(f"  Straight tiles: {sorted(set(range(9)) - CURVED_TILES)}")
print("\nTo verify during runtime:")
print("  - Check info['is_curve_tile'] after env.reset() or env.step()")
print("  - Check info['tile'] to see which tile you're on")
print("  - The observation's 3rd element (if include_curve_flag=True) will be 1.0 for curved, 0.0 for straight")

