# How Computation Delay Simulation Works

## The Problem

In real-time RL, actions are computed for one state (S₀), but by the time they're executed, the environment has evolved to a different state (S₀'). The longer the computation delay, the more outdated the action becomes.

## How It's Simulated

### Step-by-Step Flow

1. **Reset to computation state (S₀)**:
   - Teleport robot to `_computation_state_position`
   - This is where the action was computed for (from previous step)

2. **Capture as `last_pose`**:
   - Save this as the state the action was computed for
   - Used for reward/delta_t calculations

3. **Simulate computation delay**:
   - Restore previous wheel speeds (robot continues moving)
   - Let environment evolve for `step_duration` seconds
   - **During this time, state evolves from S₀ to S₀'**
   - The longer `step_duration`, the more S₀' differs from S₀

4. **Capture evolved state (S₀')**:
   - After delay, capture where the robot actually is now
   - This is the state the action will be applied to

5. **Apply action**:
   - Action was computed for S₀, but now we're in S₀'
   - **This is the key mismatch**: action computed for old state, applied to evolved state

6. **Let environment evolve with new action**:
   - Robot moves with the new action
   - Environment evolves

7. **Capture final state (S₁)**:
   - After applying action and letting it evolve
   - This is the new state

8. **Save as computation state**:
   - Save S₁ as `_computation_state_position`
   - This is where the NEXT action will be computed for

## How `step_duration` Affects Outdatedness

### Small Delay (0.05s)
- State evolves slightly during computation
- S₀' is close to S₀
- Action is still relatively appropriate
- **Performance: Good**

### Medium Delay (0.1s)
- State evolves more during computation
- S₀' differs more from S₀
- Action is less appropriate
- **Performance: Moderate**

### Large Delay (0.5s)
- State evolves significantly during computation
- S₀' is quite different from S₀
- Action is inappropriate for current state
- **Performance: Poor**

### Very Large Delay (1.0s)
- State evolves dramatically during computation
- S₀' is very different from S₀
- Action is highly inappropriate
- **Performance: Very Poor**

## Key Insight

**The longer `step_duration`, the more the state evolves during "computation", making actions more outdated.**

This is captured by:
1. Resetting to S₀ (where action was computed for)
2. Letting environment evolve for `step_duration` (simulating computation delay)
3. Applying action to S₀' (evolved state, not S₀)

The mismatch between S₀ (computation state) and S₀' (execution state) grows with `step_duration`, directly modeling the real-time RL problem.

## Example

**Step N:**
- Action a_N computed for state S_N (at end of previous step)
- `step(a_N)` called:
  - Reset to S_N
  - Let evolve for 0.5s → now in S_N'
  - Apply a_N (computed for S_N, but applied to S_N')
  - Capture S_N+1
  - Return obs_N+1

**Step N+1:**
- Action a_N+1 computed for state S_N+1
- `step(a_N+1)` called:
  - Reset to S_N+1
  - Let evolve for 0.5s → now in S_N+1'
  - Apply a_N+1 (computed for S_N+1, but applied to S_N+1')
  - ...

The 0.5s delay means the state evolves significantly between computation and execution, making actions outdated.

