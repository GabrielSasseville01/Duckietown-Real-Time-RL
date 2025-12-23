# Experiment 3: Real-Time RL with Variable Time Delay (10-run plan)

This experiment tests **robustness to variable computation delay** in gym-mode Duckiematrix, using the **action-conditioned policy** (RTAC/“real-time RL”): \(\pi(a_t \mid s_{t-1}, a_{t-1})\).

The core implementation change is that the gym environment can now **sample a different delay each step** (instead of using a fixed `step_duration`).

---

## Why variable delays, and why these distributions?

Real systems rarely have constant inference latency:
- **small jitter** around a mean (OS scheduling, minor load) → roughly symmetric noise
- **heavy tails** (cache misses, background jobs, occasional stalls) → positive-only, long right tail
- **bursty “spikes”** (GC pauses, GPU contention, I/O hiccups) → mostly normal operation with rare large delays

So we test:
- **Fixed**: control (no variability)
- **Uniform**: bounded jitter (simple, good sanity-check)
- **Truncated normal**: symmetric noise but non-negative (models “jitter”)
- **Lognormal**: positive-only heavy tail (models realistic latency distributions)
- **Mixture (“bursty”)**: rare spikes on top of a stable baseline (models real deployments best)

---

## The 10-run matrix (compute-capped, but informative)

We test **two means**:
- **0.1s** (moderate delay)
- **0.2s** (harder delay)

For each mean, we run **five variability regimes**:
1. **Fixed** at the mean
2. **Uniform, CV=0.2** (low variance)
3. **Truncated normal, CV=0.2** (low variance, symmetric)
4. **Lognormal, CV=0.6** (heavy tail, high variance)
5. **Bursty mixture** with **spike_prob=0.1** and **spike_multiplier=4.0** (rare large stalls)

Total: **2 means × 5 regimes = 10 training runs**.

### Parameter notes

- **CV** = coefficient of variation = std/mean.
- Delays are truncated to be non-negative, and bounded above to avoid pathological stalls:
  - For mean 0.1: `delay_max=0.5`
  - For mean 0.2: `delay_max=1.0`

---

## How to run

From `gym-duckiematrix/`:

```bash
python src/run_variable_delay_experiments.py \
  --means 0.1 0.2 \
  --num_episodes 400 \
  --max_steps_per_episode 2000 \
  --eval_episodes 20 \
  --eval_max_steps 2000 \
  --save_freq 100 \
  --seed 0 \
  --base_dir delay_experiments \
  --experiment_name exp3_variable_delay
```

The script creates a timestamped directory and saves:
- `results/experiment_matrix.json`: the exact 10 configurations
- `results/comparison_results.json`: evaluation metrics for each run
- per-run training metrics in `metrics/<run_name>/metrics/training_metrics.json`
- per-run evaluation metrics in `checkpoints/<run_name>/evaluation_metrics.json`

---

## Sanity-check: did the delays match what we intended?

For each run, open:
- `metrics/<run_name>/metrics/training_metrics.json`

Look at:
- `statistics.step_delays` (mean/std/p90/p99 in **ms**)

This is important because truncation/bounds can shift the realized mean slightly for heavy-tailed distributions.

---

## Optional follow-up (no extra training): generalization checks

After training, you can evaluate a trained policy under a **different** delay distribution to test robustness without extra training:

```bash
python src/sac_inference.py \
  --gym_mode --condition_on_prev_action --no_render --save_metrics \
  --policy_checkpoint <PATH_TO_POLICY> \
  --num_episodes 20 --max_steps 2000 \
  --delay_mode random --delay_dist lognormal --delay_mean 0.2 --delay_cv 0.6 --delay_max 1.0
```


