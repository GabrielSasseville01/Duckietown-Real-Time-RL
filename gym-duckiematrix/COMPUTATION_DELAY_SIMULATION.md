# Computation Delay Simulation in Gym Mode

## What It Simulates

The updated gym mode now simulates the **real-time RL problem**: actions are computed for one state, but applied after a delay when the environment has evolved to a different state.

## The Flow

### Step-by-Step Process

1. **Reset to computation state** (S₀):
   - Teleport robot to `_computation_state_position`
   - This is where the action was computed for (from previous step)

2. **Capture pose as last_pose**:
   - Save this as the state the action was computed for
   - Used for reward calculation and delta_t

3. **Apply action**:
   - Set wheel speeds from the action
   - Action was computed for S₀, now we apply it

4. **Simulate delay** (`step_duration`):
   - Sleep for `step_duration` seconds
   - This simulates: computation delay + action execution time
   - In real-time, the environment would evolve during this time

5. **Capture new pose** (S₁):
   - After delay, capture where the robot is now
   - This is the new state after applying action

6. **Save as computation state**:
   - Save new position as `_computation_state_position`
   - This is where the NEXT action will be computed for

7. **Return observation**:
   - Return observation of S₁
   - Agent will compute next action for S₁

## How It Matches Real-Time RL

### Real-Time Scenario
```
t₀: Observe S₀ → Compute a₀ (takes Δt)
t₀+Δt: Execute a₀ (but state is now S₀')
t₁: Observe S₁ → Compute a₁
```

### Gym Mode Simulation
```
Step N:
  - Reset to S₀ (where a₀ was computed for)
  - Apply a₀
  - Wait step_duration (simulates Δt)
  - Capture S₁ (new state)
  - Save S₁ as computation state for next step

Step N+1:
  - Reset to S₁ (where a₁ was computed for)
  - Apply a₁
  - ...
```

## Key Insight

**The action is always applied to the state it was computed for**, but:
- There's a delay between computation and execution
- During that delay, the environment would evolve (in real-time)
- The delay is simulated by `step_duration`

This creates a mismatch: action computed for S₀, but by the time it's applied, we're in a different situation.

## What `step_duration` Represents

- **0.05s**: Fast computer (50ms computation time)
- **0.1s**: Medium computer (100ms computation time)
- **0.5s**: Slow computer (500ms computation time)
- **1.0s**: Very slow computer (1s computation time)

Longer delays = more state evolution during computation = worse performance

## Comparison to Real-Time Mode

**Real-Time Mode** (`DB21J.py`):
- Environment runs continuously
- Action computation happens while environment evolves
- Natural delay between observation and action

**Gym Mode** (`DB21J_gym.py`):
- Controlled simulation with explicit delay
- Teleports to simulate "action computed for this state"
- `step_duration` controls the delay magnitude

Both simulate the real-time RL problem, but gym mode gives you **controlled, reproducible delays** for experiments.

