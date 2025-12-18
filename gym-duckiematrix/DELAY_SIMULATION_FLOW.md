# Delay Simulation Flow - Detailed Explanation

## The Real-Time RL Problem

**In real-time:**
1. t₀: Observe state S₀, start computing action a₀
2. t₀ to t₀+delay: Computing a₀ (robot continues with previous action a₋₁)
3. t₀+delay: Action a₀ ready, but robot is now in state S₀' (evolved from S₀)
4. t₀+delay: Apply a₀ to S₀' (not S₀!)
5. t₁: Observe new state S₁

## How We Simulate It

### Step-by-Step Flow

**Step N (in gym mode):**

1. **Reset to computation state (S₀)**:
   - Teleport robot to `_computation_state_position`
   - This is where action a_N was computed for (from previous step)

2. **Capture S₀ as `last_pose`**:
   - Save this as the state the action was computed for
   - Used for reward/delta_t calculations

3. **Restore previous wheel speeds**:
   - Set motors to `_last_wheel_speeds` (previous action a₋₁)
   - Robot continues moving with the previous action

4. **Simulate computation delay**:
   - Sleep for `step_duration` seconds
   - **During this time, robot moves with previous action**
   - State evolves from S₀ → S₀'
   - The longer the delay, the more S₀' differs from S₀

5. **Capture evolved state (S₀')**:
   - After delay, capture where robot actually is
   - This is the state the action will be applied to
   - **We're already at S₀', no need to reset again**

6. **Apply new action (a_N)**:
   - Action was computed for S₀, but we're now in S₀'
   - Set motors to new action values
   - **This is the key mismatch: action for S₀, applied to S₀'**

7. **Let environment evolve with new action**:
   - Small sleep to let new action take effect
   - Robot moves with new action

8. **Capture final state (S₁)**:
   - After applying action and letting it evolve
   - This is the new state

9. **Save S₁ as computation state**:
   - Save as `_computation_state_position`
   - This is where the NEXT action (a_N+1) will be computed for

## Key Points

### ✅ What We Do

- **Continue with previous action during delay**: Yes! This simulates real-time behavior
- **Capture evolved state**: Yes! We capture S₀' after the delay
- **Apply action to evolved state**: Yes! Action computed for S₀, applied to S₀'

### ❌ What We DON'T Do

- **Reset to S₀' after capturing it**: No need! We're already there
- **Apply action to S₀**: No! That would be wrong - we apply to S₀'

## Why This Works

The flow correctly simulates:
1. **State evolution during computation**: Robot continues with previous action
2. **Outdated state problem**: Action computed for S₀, but applied to S₀'
3. **Delay effect**: Longer delays → more state evolution → more outdated actions

## Example with step_duration = 0.5s

**Step N:**
- Reset to S_N (where a_N was computed for)
- Restore previous wheel speeds (robot continues with a_N-1)
- Sleep 0.5s (robot moves, state evolves to S_N')
- Capture S_N' (robot is here now)
- Apply a_N (computed for S_N, but applied to S_N')
- Sleep a bit (robot moves with a_N)
- Capture S_N+1 (new state)
- Save S_N+1 as computation state

**Step N+1:**
- Reset to S_N+1 (where a_N+1 was computed for)
- Restore previous wheel speeds (robot continues with a_N)
- Sleep 0.5s (robot moves, state evolves to S_N+1')
- Capture S_N+1' (robot is here now)
- Apply a_N+1 (computed for S_N+1, but applied to S_N+1')
- ...

## The Mismatch

The longer `step_duration`, the more:
- S₀' differs from S₀ (more state evolution)
- Action becomes outdated (computed for wrong state)
- Performance degrades

This is exactly what we want to measure!

