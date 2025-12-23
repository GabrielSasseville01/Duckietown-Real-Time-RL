"""
Experiment 3 runner: Real-Time RL (action conditioning) under VARIABLE computation delays.

This script trains ~10 agents (configurable) where the gym-mode delay is sampled per-step
from different distributions, then evaluates each trained agent under the same delay setting.

Rationale:
- Keep compute bounded (≈10 training runs)
- Cover two mean delays (0.1s, 0.2s) and multiple "shapes" of variability:
  fixed, uniform jitter, truncated normal jitter, heavy-tailed lognormal, and bursty mixture.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _this_dir() -> Path:
    return Path(__file__).resolve().parent


def _gym_duckiematrix_root() -> Path:
    # This file lives in gym-duckiematrix/src/
    return _this_dir().parent


def _resolve_under_root(path_str: str) -> Path:
    """
    Resolve a user-provided path.
    - If absolute: keep it
    - If relative: interpret relative to gym-duckiematrix/ (NOT the current working directory)
    This makes it safe to run from either gym-duckiematrix/ or gym-duckiematrix/src/.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (_gym_duckiematrix_root() / p).resolve()


def _python_exe() -> str:
    # Use the current interpreter (e.g., your activated .venv) for subprocess calls
    return sys.executable or "python3"


def find_latest_checkpoint(checkpoint_dir: str):
    """Find the latest checkpoint in a directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None, 0

    policy_files = list(checkpoint_dir.glob("sac_policy_ep*.pth"))
    if not policy_files:
        return None, 0

    episodes = []
    for f in policy_files:
        try:
            ep_num = int(f.stem.split("_ep")[1])
            episodes.append((ep_num, f))
        except (IndexError, ValueError):
            continue

    if not episodes:
        return None, 0

    latest_ep, latest_file = max(episodes, key=lambda x: x[0])
    return str(latest_file), latest_ep


def run_training_experiment(
    experiment_name: str,
    num_episodes: int,
    max_steps_per_episode: int,
    checkpoint_dir: str,
    metrics_dir: str,
    save_freq: int,
    batch_size: int,
    update_freq: int,
    resume_from_existing: bool,
    seed: int | None,
    # Gym env delay config
    step_duration: float,
    delay_mode: str,
    delay_dist: str | None = None,
    delay_mean: float | None = None,
    delay_cv: float | None = None,
    delay_std: float | None = None,
    delay_min: float | None = None,
    delay_max: float | None = None,
    delay_spike_prob: float | None = None,
    delay_spike_multiplier: float | None = None,
):
    """Train one agent under a specific variable-delay configuration."""
    start_episode = 0
    policy_checkpoint = None
    q1_checkpoint = None
    q2_checkpoint = None

    if resume_from_existing:
        latest_policy, latest_ep = find_latest_checkpoint(checkpoint_dir)
        if latest_policy:
            start_episode = latest_ep
            policy_checkpoint = latest_policy
            checkpoint_path = Path(latest_policy)
            q1_path = checkpoint_path.parent / f"sac_q1_ep{latest_ep}.pth"
            q2_path = checkpoint_path.parent / f"sac_q2_ep{latest_ep}.pth"
            if q1_path.exists():
                q1_checkpoint = str(q1_path)
            if q2_path.exists():
                q2_checkpoint = str(q2_path)

            final_checkpoint = Path(checkpoint_dir) / "sac_policy_final.pth"
            if final_checkpoint.exists() or latest_ep >= num_episodes:
                print(f"\n{'='*60}")
                print(f"Skipping: {experiment_name} (already complete at episode {latest_ep})")
                print(f"{'='*60}\n")
                return str(final_checkpoint) if final_checkpoint.exists() else latest_policy

    remaining_episodes = num_episodes - start_episode
    if remaining_episodes <= 0:
        final_checkpoint = Path(checkpoint_dir) / "sac_policy_final.pth"
        return str(final_checkpoint) if final_checkpoint.exists() else None

    print(f"\n{'='*60}")
    print(f"Training: {experiment_name}")
    print(f"  gym_mode: True")
    print(f"  action-conditioning: ENABLED")
    print(f"  step_duration (legacy/fallback): {step_duration}s")
    print(f"  delay_mode: {delay_mode}")
    if delay_mode == "random":
        print(f"  delay_dist: {delay_dist}")
        print(f"  delay_mean: {delay_mean}")
        print(f"  delay_cv: {delay_cv}")
        print(f"  delay_std: {delay_std}")
        print(f"  delay_min: {delay_min}")
        print(f"  delay_max: {delay_max}")
        print(f"  spike_prob: {delay_spike_prob} | spike_multiplier: {delay_spike_multiplier}")
    if seed is not None:
        print(f"  seed: {seed}")
    if start_episode > 0:
        print(f"  Resuming from episode {start_episode} (remaining {remaining_episodes})")
    print(f"{'='*60}\n")

    sac_agent_path = (_this_dir() / "sac_agent.py").resolve()
    cmd = [
        _python_exe(), str(sac_agent_path),
        "--gym_mode",
        "--condition_on_prev_action",
        "--num_episodes", str(remaining_episodes),
        "--max_steps_per_episode", str(max_steps_per_episode),
        "--batch_size", str(batch_size),
        "--update_freq", str(update_freq),
        "--save_freq", str(save_freq),
        "--checkpoint_dir", checkpoint_dir,
        "--metrics_dir", metrics_dir,
        "--start_episode", str(start_episode),
        "--step_duration", str(step_duration),
        "--delay_mode", delay_mode,
    ]

    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    if delay_mode == "random":
        if delay_dist is not None:
            cmd.extend(["--delay_dist", str(delay_dist)])
        if delay_mean is not None:
            cmd.extend(["--delay_mean", str(delay_mean)])
        if delay_cv is not None:
            cmd.extend(["--delay_cv", str(delay_cv)])
        if delay_std is not None:
            cmd.extend(["--delay_std", str(delay_std)])
        if delay_min is not None:
            cmd.extend(["--delay_min", str(delay_min)])
        if delay_max is not None:
            cmd.extend(["--delay_max", str(delay_max)])
        if delay_spike_prob is not None:
            cmd.extend(["--delay_spike_prob", str(delay_spike_prob)])
        if delay_spike_multiplier is not None:
            cmd.extend(["--delay_spike_multiplier", str(delay_spike_multiplier)])

    if policy_checkpoint:
        cmd.extend(["--policy_checkpoint", policy_checkpoint])
    if q1_checkpoint:
        cmd.extend(["--q1_checkpoint", q1_checkpoint])
    if q2_checkpoint:
        cmd.extend(["--q2_checkpoint", q2_checkpoint])

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: Training failed for {experiment_name}")
        return None

    final_checkpoint = os.path.join(checkpoint_dir, "sac_policy_final.pth")
    return final_checkpoint if os.path.exists(final_checkpoint) else None


def run_evaluation(
    experiment_name: str,
    checkpoint_path: str,
    step_duration: float,
    num_episodes: int,
    max_steps: int,
    seed: int | None,
    # Delay config (must match training to avoid distribution shift unless desired)
    delay_mode: str,
    delay_dist: str | None = None,
    delay_mean: float | None = None,
    delay_cv: float | None = None,
    delay_std: float | None = None,
    delay_min: float | None = None,
    delay_max: float | None = None,
    delay_spike_prob: float | None = None,
    delay_spike_multiplier: float | None = None,
):
    print(f"\nEvaluating: {experiment_name}")

    sac_inference_path = (_this_dir() / "sac_inference.py").resolve()
    cmd = [
        _python_exe(), str(sac_inference_path),
        "--policy_checkpoint", checkpoint_path,
        "--num_episodes", str(num_episodes),
        "--max_steps", str(max_steps),
        "--no_render",
        "--save_metrics",
        "--gym_mode",
        "--condition_on_prev_action",
        "--step_duration", str(step_duration),
        "--delay_mode", delay_mode,
    ]

    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    if delay_mode == "random":
        if delay_dist is not None:
            cmd.extend(["--delay_dist", str(delay_dist)])
        if delay_mean is not None:
            cmd.extend(["--delay_mean", str(delay_mean)])
        if delay_cv is not None:
            cmd.extend(["--delay_cv", str(delay_cv)])
        if delay_std is not None:
            cmd.extend(["--delay_std", str(delay_std)])
        if delay_min is not None:
            cmd.extend(["--delay_min", str(delay_min)])
        if delay_max is not None:
            cmd.extend(["--delay_max", str(delay_max)])
        if delay_spike_prob is not None:
            cmd.extend(["--delay_spike_prob", str(delay_spike_prob)])
        if delay_spike_multiplier is not None:
            cmd.extend(["--delay_spike_multiplier", str(delay_spike_multiplier)])

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: Evaluation failed for {experiment_name}")
        return None

    metrics_file = Path(checkpoint_path).parent / "evaluation_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            eval_metrics = json.load(f)
    else:
        eval_metrics = {"avg_reward": None, "avg_length": None, "note": "Metrics JSON missing"}

    metrics = {
        "experiment_name": experiment_name,
        "use_gym_mode": True,
        "condition_on_prev_action": True,
        "step_duration": step_duration,
        "delay_mode": delay_mode,
        "delay_dist": delay_dist if delay_mode == "random" else "fixed",
        "delay_mean": delay_mean,
        "delay_cv": delay_cv,
        "delay_std": delay_std,
        "delay_min": delay_min,
        "delay_max": delay_max,
        "delay_spike_prob": delay_spike_prob,
        "delay_spike_multiplier": delay_spike_multiplier,
        "seed": seed,
        "checkpoint": checkpoint_path,
        **eval_metrics,
    }
    return metrics


def build_default_matrix(means: list[float]) -> list[dict]:
    """
    2 means x 5 variability regimes = 10 runs.

    We explicitly include a fixed-delay control at each mean to separate:
    - effect of the mean delay itself, vs
    - additional penalty from variability / heavy tails / spikes.
    """
    matrix = []
    for m in means:
        # Conservative upper bounds to avoid pathological stalls while still allowing tails to matter.
        delay_max = 0.5 if m <= 0.12 else 1.0

        matrix.extend([
            {
                "name": f"mean{m:.3f}_fixed",
                "step_duration": m,
                "delay_mode": "fixed",
            },
            {
                "name": f"mean{m:.3f}_uniform_cv0p2",
                "step_duration": m,
                "delay_mode": "random",
                "delay_dist": "uniform",
                "delay_mean": m,
                "delay_cv": 0.2,
                "delay_min": 0.0,
                "delay_max": delay_max,
            },
            {
                "name": f"mean{m:.3f}_normal_cv0p2",
                "step_duration": m,
                "delay_mode": "random",
                "delay_dist": "normal",
                "delay_mean": m,
                "delay_cv": 0.2,
                "delay_min": 0.0,
                "delay_max": delay_max,
            },
            {
                "name": f"mean{m:.3f}_lognormal_cv0p6",
                "step_duration": m,
                "delay_mode": "random",
                "delay_dist": "lognormal",
                "delay_mean": m,
                "delay_cv": 0.6,
                "delay_min": 0.0,
                "delay_max": delay_max,
            },
            {
                "name": f"mean{m:.3f}_bursty_mix_p0p1_x4",
                "step_duration": m,
                "delay_mode": "random",
                "delay_dist": "mixture",
                "delay_mean": m,
                "delay_cv": 0.6,  # spike variability proxy
                "delay_min": 0.0,
                "delay_max": delay_max,
                "delay_spike_prob": 0.1,
                "delay_spike_multiplier": 4.0,
            },
        ])
    return matrix


def run_variable_delay_experiments(
    means: list[float],
    num_episodes: int,
    max_steps_per_episode: int,
    eval_episodes: int,
    eval_max_steps: int,
    base_dir: str,
    experiment_name: str | None,
    resume_experiment: str | None,
    seed: int | None,
    batch_size: int,
    update_freq: int,
    save_freq: int,
):
    if resume_experiment:
        exp_dir = _resolve_under_root(resume_experiment)
        if not exp_dir.exists():
            raise ValueError(f"Experiment directory does not exist: {resume_experiment}")
        print(f"\n{'='*60}\nRESUMING EXPERIMENT 3\nDirectory: {exp_dir}\n{'='*60}\n")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir_path = _resolve_under_root(base_dir)
        if experiment_name:
            exp_dir = base_dir_path / f"{experiment_name}_{timestamp}"
        else:
            exp_dir = base_dir_path / f"experiment3_variable_delay_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)

    results_dir = exp_dir / "results"
    results_dir.mkdir(exist_ok=True)

    matrix = build_default_matrix(means)
    matrix_file = results_dir / "experiment_matrix.json"
    with open(matrix_file, "w") as f:
        json.dump(matrix, f, indent=2)

    print(f"\n{'='*60}")
    print("EXPERIMENT 3: VARIABLE DELAY SUITE")
    print(f"{'='*60}")
    print(f"Directory: {exp_dir}")
    print(f"Runs: {len(matrix)} (target ≈10)")
    print(f"Means: {means}")
    print(f"Training episodes/run: {num_episodes}")
    print(f"Evaluation episodes/run: {eval_episodes}")
    if seed is not None:
        print(f"Seed: {seed}")
    print(f"Matrix saved to: {matrix_file}")
    print(f"{'='*60}\n")

    all_results = []
    for cfg in matrix:
        exp_name = cfg["name"]
        checkpoint_dir = str(exp_dir / "checkpoints" / exp_name)
        metrics_dir = str(exp_dir / "metrics" / exp_name)

        checkpoint = run_training_experiment(
            experiment_name=exp_name,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            checkpoint_dir=checkpoint_dir,
            metrics_dir=metrics_dir,
            save_freq=save_freq,
            batch_size=batch_size,
            update_freq=update_freq,
            resume_from_existing=resume_experiment is not None,
            seed=seed,
            step_duration=cfg.get("step_duration"),
            delay_mode=cfg.get("delay_mode"),
            delay_dist=cfg.get("delay_dist"),
            delay_mean=cfg.get("delay_mean"),
            delay_cv=cfg.get("delay_cv"),
            delay_std=cfg.get("delay_std"),
            delay_min=cfg.get("delay_min"),
            delay_max=cfg.get("delay_max"),
            delay_spike_prob=cfg.get("delay_spike_prob"),
            delay_spike_multiplier=cfg.get("delay_spike_multiplier"),
        )

        if checkpoint:
            eval_results = run_evaluation(
                experiment_name=exp_name,
                checkpoint_path=checkpoint,
                step_duration=cfg.get("step_duration"),
                num_episodes=eval_episodes,
                max_steps=eval_max_steps,
                seed=seed,
                delay_mode=cfg.get("delay_mode"),
                delay_dist=cfg.get("delay_dist"),
                delay_mean=cfg.get("delay_mean"),
                delay_cv=cfg.get("delay_cv"),
                delay_std=cfg.get("delay_std"),
                delay_min=cfg.get("delay_min"),
                delay_max=cfg.get("delay_max"),
                delay_spike_prob=cfg.get("delay_spike_prob"),
                delay_spike_multiplier=cfg.get("delay_spike_multiplier"),
            )
            if eval_results:
                all_results.append(eval_results)

        results_file = results_dir / "comparison_results.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Experiment 3 completed!")
    print(f"Results saved to: {results_dir / 'comparison_results.json'}")
    print(f"{'='*60}\n")
    return all_results, str(exp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 3: variable delay real-time RL experiments")
    parser.add_argument("--means", type=float, nargs="+", default=[0.1, 0.2],
                        help="Mean delays (seconds) to test (default: 0.1 0.2)")
    parser.add_argument("--num_episodes", type=int, default=400,
                        help="Training episodes per run (default: 400)")
    parser.add_argument("--max_steps_per_episode", type=int, default=2000,
                        help="Max steps per training episode (default: 2000)")
    parser.add_argument("--eval_episodes", type=int, default=20,
                        help="Evaluation episodes per run (default: 20)")
    parser.add_argument("--eval_max_steps", type=int, default=2000,
                        help="Max steps per evaluation episode (default: 2000)")
    parser.add_argument("--base_dir", type=str, default="delay_experiments",
                        help="Base directory for experiments (default: delay_experiments)")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Optional name prefix for this run (default: timestamped)")
    parser.add_argument("--resume_experiment", type=str, default=None,
                        help="Resume from an existing experiment directory")
    parser.add_argument("--seed", type=int, default=0,
                        help="Global seed for reproducibility (default: 0)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Training batch size (default: 256)")
    parser.add_argument("--update_freq", type=int, default=1,
                        help="Update frequency (default: 1)")
    parser.add_argument("--save_freq", type=int, default=100,
                        help="Checkpoint save frequency in episodes (default: 100)")

    args = parser.parse_args()

    results, exp_dir = run_variable_delay_experiments(
        means=args.means,
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
        base_dir=args.base_dir,
        experiment_name=args.experiment_name,
        resume_experiment=args.resume_experiment,
        seed=args.seed,
        batch_size=args.batch_size,
        update_freq=args.update_freq,
        save_freq=args.save_freq,
    )

    print(f"\nExperiment directory: {exp_dir}")

