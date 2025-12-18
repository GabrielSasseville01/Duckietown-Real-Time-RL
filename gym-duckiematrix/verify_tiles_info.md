# How to Verify CURVED_TILES = {0, 2, 6, 8}

## Current Definition
In `src/gym_duckiematrix/DB21J.py` line 25:
```python
CURVED_TILES = {0, 2, 6, 8}
```

This means:
- **Curved tiles**: 0, 2, 6, 8
- **Straight tiles**: 1, 3, 4, 5, 7

## How to Verify During Runtime

### Method 1: Check the info dict
After `env.reset()` or `env.step()`, check:
```python
obs, info = env.reset()
print(f"Tile: {info['tile']}, Is curve: {info['is_curve_tile']}")
```

### Method 2: Check the observation
If `include_curve_flag=True`, the 3rd element of the observation is:
- `1.0` for curved tiles
- `0.0` for straight tiles

```python
obs, info = env.reset()
if len(obs) == 3:
    print(f"Tile: {info['tile']}, Curve flag: {obs[2]}")
```

### Method 3: Add temporary print in SAC training
Add this in the training loop after `env.reset()`:
```python
obs, info = env.reset()
if episode % 10 == 0:  # Print every 10 episodes
    print(f"Episode {episode}: Tile {info['tile']}, is_curve={info['is_curve_tile']}")
```

## Expected Results
If the definition is correct, you should see:
- Tiles 0, 2, 6, 8 → `is_curve_tile=True`, `obs[2]=1.0`
- Tiles 1, 3, 4, 5, 7 → `is_curve_tile=False`, `obs[2]=0.0`

