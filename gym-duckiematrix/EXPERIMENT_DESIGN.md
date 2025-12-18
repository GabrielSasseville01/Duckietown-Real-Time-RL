# Experiment Design: Real-Time RL vs Gym Mode

## Research Question

**How does computation delay affect RL agent performance in real-world robotics scenarios?**

## Current Implementation

### What `step_duration` Currently Does

In gym mode, `step_duration` controls:
- How long the simulation runs after setting wheel speeds
- This simulates: "If it takes X seconds to compute an action, how does the robot perform?"

**Current flow**:
1. Observe state S₀
2. Compute action a₀ (instantaneous in simulation)
3. Execute a₀
4. Let environment evolve for `step_duration` seconds
5. Observe new state S₁

### What It Should Simulate

**Ideal flow** (to match real-time RL problem):
1. Observe state S₀ at time t₀
2. **Simulate computation delay**: Wait `computation_delay` seconds
   - During this time, environment evolves (robot keeps moving)
   - State becomes S₀' (different from S₀)
3. Execute action a₀ (computed for S₀, but applied to S₀')
4. Observe new state S₁

## The Difference

### Current Implementation
- Action is computed instantly
- Action is applied immediately
- Environment evolves for `step_duration`
- **This is more like "action duration" than "computation delay"**

### What We Want
- Action computation takes time (simulated delay)
- Environment evolves during computation
- Action (computed for old state) is applied to new state
- **This matches the real-time RL problem**

## Proposed Improvement

To better simulate real-time delays, we could modify gym mode to:

```python
def step(self, actions):
    # 1. Observe state S₀
    obs_state = self.robot.pose.capture()
    
    # 2. SIMULATE COMPUTATION DELAY
    # During this time, environment evolves (robot keeps moving with previous action)
    time.sleep(self.computation_delay)
    
    # 3. Capture current state S₀' (may be different from S₀)
    current_state = self.robot.pose.capture()
    
    # 4. Execute action (computed for S₀, but applied to S₀')
    self.robot.motors.set_pwm(left=actions[0], right=actions[1])
    
    # 5. Let environment evolve
    time.sleep(self.step_duration)
    
    # 6. Observe new state S₁
    new_state = self.robot.pose.capture()
    
    return obs, reward, done, info
```

## Experiment Interpretation

### Current Setup (What You Have)

Your experiment with varying `step_duration` measures:
- **"How does performance change when actions take longer to execute?"**
- This is still valuable! It shows robustness to slower control loops

### Improved Setup (What Could Be Better)

With computation delay simulation:
- **"How does performance degrade when there's delay between observation and action?"**
- This directly matches the real-time RL problem

## Real-World Implications

### Current Experiment
Answers: "If my control loop runs at 10Hz vs 1Hz, how much does performance degrade?"

### Improved Experiment
Answers: "If my neural network takes 50ms vs 500ms to compute, how much does performance degrade?"

## Recommendation

Your current experiment is **still valid and valuable**! It measures:
- Control frequency impact
- Action duration effects
- System responsiveness requirements

To make it more aligned with the paper, you could:
1. **Keep current experiment** (measures control frequency)
2. **Add computation delay simulation** (measures observation-to-action delay)
3. **Compare both** to see which has bigger impact

## Key Takeaway

The paper's insight is that **delays matter**. Your experiment shows:
- Real-time (no artificial delay): Best performance
- Gym mode with delays: Performance degrades
- Longer delays: Worse performance

This validates the paper's claim that delays hurt performance, and your experiment quantifies **how much** they hurt!

