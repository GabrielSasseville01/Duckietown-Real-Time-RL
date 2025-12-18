"""
Script to verify which tiles are curved vs straight in the Duckietown loop map.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gym_duckiematrix.DB21J import DuckiematrixDB21JEnv
from duckietown.sdk.utils.loop_lane_position import get_closest_tile, perfect_initial_position
import time

print("Verifying curved tiles...")
print("=" * 50)

# Create environment with curve flag enabled
env = DuckiematrixDB21JEnv(entity_name="map_0/vehicle_0", include_curve_flag=True)

# Test each tile from 0 to 8
tile_results = {}
for tile_id in range(9):
    try:
        # Reset to perfect position in this tile
        position = perfect_initial_position(tile=tile_id, position_along_tile=0.5)
        obs, info = env.reset(position=position)
        
        # Get tile info from the environment
        reported_tile = info.get("tile")
        is_curve = info.get("is_curve_tile", False)
        
        # Also check the observation if curve flag is enabled
        curve_flag_from_obs = None
        if len(obs) == 3:
            curve_flag_from_obs = obs[2]
        
        tile_results[tile_id] = {
            "reported_tile": reported_tile,
            "is_curve_tile": is_curve,
            "curve_flag_obs": curve_flag_from_obs,
            "position": position
        }
        
        print(f"Tile {tile_id}: is_curve={is_curve}, reported_tile={reported_tile}, obs_flag={curve_flag_from_obs}")
        
        time.sleep(0.1)  # Small delay
        
    except Exception as e:
        print(f"Tile {tile_id}: ERROR - {e}")
        tile_results[tile_id] = {"error": str(e)}

print("\n" + "=" * 50)
print("Summary:")
print("=" * 50)

curved_found = []
straight_found = []

for tile_id, result in tile_results.items():
    if "error" not in result:
        if result["is_curve_tile"]:
            curved_found.append(tile_id)
        else:
            straight_found.append(tile_id)

print(f"Curved tiles found: {curved_found}")
print(f"Straight tiles found: {straight_found}")
print(f"\nCurrent CURVED_TILES definition: {{0, 2, 6, 8}}")
print(f"Matches found tiles: {set(curved_found) == {0, 2, 6, 8}}")

env.robot.motors.stop()

