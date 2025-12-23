"""
Plotting utilities for Experiment 3 (variable delay).

Inputs:
  - comparison_results.json (produced by run_variable_delay_experiments.py)
  - training_metrics.json for each run (to read realized delay stats: p90/p99, etc.)

Outputs (PNG) into the same directory as comparison_results.json unless --out_dir is set.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt


REGIME_ORDER = [
    "fixed",
    "uniform",
    "normal",
    "lognormal",
    "mixture",
]

REGIME_LABEL = {
    "fixed": "Fixed",
    "uniform": "Uniform (CV=0.2)",
    "normal": "TruncNormal (CV=0.2)",
    "lognormal": "Lognormal (CV=0.6)",
    "mixture": "Bursty mix (p=0.1, ×4)",
}


@dataclass
class RunRow:
    name: str
    mean_s: float
    dist: str
    avg_reward: float
    reward_std: float
    episode_rewards: list[float]
    checkpoint_path: Path
    # realized delay stats from training (ms)
    delay_mean_ms: Optional[float] = None
    delay_p50_ms: Optional[float] = None
    delay_p90_ms: Optional[float] = None
    delay_p99_ms: Optional[float] = None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _infer_mean_from_row(d: dict) -> float:
    # For fixed, delay_mean is null in comparison_results; use step_duration as the mean proxy.
    if d.get("delay_mode") == "fixed":
        return float(d["step_duration"])
    return float(d["delay_mean"])


def _infer_dist_from_row(d: dict) -> str:
    if d.get("delay_mode") == "fixed":
        return "fixed"
    return str(d.get("delay_dist", "unknown")).lower()


def _training_metrics_path_from_checkpoint(checkpoint_path: Path) -> Path:
    """
    checkpoint_path:
      .../delay_experiments/<exp>/checkpoints/<run>/sac_policy_final.pth
    metrics:
      .../delay_experiments/<exp>/metrics/<run>/metrics/training_metrics.json
    """
    run_dir = checkpoint_path.parent.name
    exp_dir = checkpoint_path.parent.parent.parent  # .../<exp>/
    return exp_dir / "metrics" / run_dir / "metrics" / "training_metrics.json"


def _load_delay_stats_ms(training_metrics_path: Path) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not training_metrics_path.exists():
        return None, None, None, None
    with open(training_metrics_path, "r") as f:
        tm = json.load(f)
    stats = tm.get("statistics", {}).get("step_delays", {})
    return (
        _safe_float(stats.get("mean_ms")),
        _safe_float(stats.get("p50_ms")),
        _safe_float(stats.get("p90_ms")),
        _safe_float(stats.get("p99_ms")),
    )


def load_runs(results_file: Path) -> list[RunRow]:
    with open(results_file, "r") as f:
        data = json.load(f)

    rows: list[RunRow] = []
    for d in data:
        checkpoint = Path(d["checkpoint"])
        mean_s = _infer_mean_from_row(d)
        dist = _infer_dist_from_row(d)
        avg_reward = float(d["avg_reward"])
        reward_std = float(d.get("reward_std", 0.0))
        episode_rewards = list(map(float, d.get("episode_rewards", [])))

        tm_path = _training_metrics_path_from_checkpoint(checkpoint)
        delay_mean_ms, delay_p50_ms, delay_p90_ms, delay_p99_ms = _load_delay_stats_ms(tm_path)

        rows.append(
            RunRow(
                name=str(d["experiment_name"]),
                mean_s=mean_s,
                dist=dist,
                avg_reward=avg_reward,
                reward_std=reward_std,
                episode_rewards=episode_rewards,
                checkpoint_path=checkpoint,
                delay_mean_ms=delay_mean_ms,
                delay_p50_ms=delay_p50_ms,
                delay_p90_ms=delay_p90_ms,
                delay_p99_ms=delay_p99_ms,
            )
        )
    return rows


def _sorted_by_regime(rows: list[RunRow]) -> list[RunRow]:
    idx = {k: i for i, k in enumerate(REGIME_ORDER)}
    return sorted(rows, key=lambda r: idx.get(r.dist, 999))

def load_runs_multi(results_files: list[Path]) -> list[RunRow]:
    """
    Load runs from multiple experiment directories so we can plot across means.
    If duplicates exist for (mean, dist), we keep the first one encountered.
    """
    rows: list[RunRow] = []
    seen: set[tuple[float, str]] = set()
    for rf in results_files:
        for r in load_runs(rf):
            key = (r.mean_s, r.dist)
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    return rows


def plot_reward_vs_mean_by_regime(rows: list[RunRow], out_dir: Path) -> None:
    """
    Combined plot across means:
      x-axis: mean delay (seconds)
      y-axis: evaluation avg reward
      one line per delay regime (fixed/uniform/normal/lognormal/mixture)
    """
    # group by regime
    by_dist: dict[str, list[RunRow]] = {}
    for r in rows:
        by_dist.setdefault(r.dist, []).append(r)

    fig, ax = plt.subplots(figsize=(8, 5))

    color = {
        "fixed": "black",
        "uniform": "#1f77b4",
        "normal": "#2ca02c",
        "lognormal": "#ff7f0e",
        "mixture": "#d62728",
    }

    for dist in REGIME_ORDER:
        if dist not in by_dist:
            continue
        pts = sorted(by_dist[dist], key=lambda r: r.mean_s)
        xs = [p.mean_s for p in pts]
        ys = [p.avg_reward for p in pts]
        es = [p.reward_std for p in pts]
        ax.errorbar(
            xs,
            ys,
            yerr=es,
            marker="o",
            linewidth=2,
            capsize=4,
            label=REGIME_LABEL.get(dist, dist),
            color=color.get(dist, None),
        )

    ax.set_xlabel("Mean delay (seconds)")
    ax.set_ylabel("Evaluation avg reward (20 episodes)")
    ax.set_title("Experiment 3: performance vs mean delay (lines = delay regime)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fname = out_dir / "exp3_reward_vs_mean_by_regime.png"
    fig.savefig(fname, dpi=180)
    plt.close(fig)


def plot_degradation_vs_fixed_across_means(rows: list[RunRow], out_dir: Path) -> None:
    """
    Combined plot across means:
      y-axis: percent change vs fixed baseline at same mean
      x-axis: mean delay
      one line per variable-delay regime (excluding fixed)

    Note: percent changes can blow up if the fixed baseline reward at some mean is near zero.
    """
    means = sorted({r.mean_s for r in rows})
    fixed_by_mean = {
        m: next((r for r in rows if abs(r.mean_s - m) < 1e-9 and r.dist == "fixed"), None)
        for m in means
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for dist in ["uniform", "normal", "lognormal", "mixture"]:
        xs: list[float] = []
        ys: list[float] = []
        for m in means:
            fixed = fixed_by_mean.get(m)
            if fixed is None:
                continue
            variant = next((r for r in rows if abs(r.mean_s - m) < 1e-9 and r.dist == dist), None)
            if variant is None:
                continue
            pct = 100.0 * (variant.avg_reward / fixed.avg_reward - 1.0)
            xs.append(m)
            ys.append(pct)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, label=REGIME_LABEL.get(dist, dist))

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Mean delay (seconds)")
    ax.set_ylabel("Δ reward vs fixed baseline (%)")
    ax.set_title("Experiment 3: degradation vs fixed across means (lines = delay regime)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fname = out_dir / "exp3_degradation_vs_fixed_across_means.png"
    fig.savefig(fname, dpi=180)
    plt.close(fig)


def plot_abs_delta_vs_fixed_across_means(rows: list[RunRow], out_dir: Path) -> None:
    """
    Same as plot_degradation_vs_fixed_across_means, but using absolute reward difference:
      Δ = avg_reward(variant) - avg_reward(fixed at same mean)
    This is stable even when the fixed baseline is near 0.
    """
    means = sorted({r.mean_s for r in rows})
    fixed_by_mean = {
        m: next((r for r in rows if abs(r.mean_s - m) < 1e-9 and r.dist == "fixed"), None)
        for m in means
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for dist in ["uniform", "normal", "lognormal", "mixture"]:
        xs: list[float] = []
        ys: list[float] = []
        for m in means:
            fixed = fixed_by_mean.get(m)
            if fixed is None:
                continue
            variant = next((r for r in rows if abs(r.mean_s - m) < 1e-9 and r.dist == dist), None)
            if variant is None:
                continue
            xs.append(m)
            ys.append(variant.avg_reward - fixed.avg_reward)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, label=REGIME_LABEL.get(dist, dist))

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Mean delay (seconds)")
    ax.set_ylabel("Δ reward vs fixed baseline (absolute)")
    ax.set_title("Experiment 3: absolute Δ vs fixed across means (lines = delay regime)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fname = out_dir / "exp3_abs_delta_vs_fixed_across_means.png"
    fig.savefig(fname, dpi=180)
    plt.close(fig)


def plot_bars_per_mean(rows: list[RunRow], out_dir: Path) -> None:
    means = sorted({r.mean_s for r in rows})
    for m in means:
        group = _sorted_by_regime([r for r in rows if abs(r.mean_s - m) < 1e-9])
        labels = [REGIME_LABEL.get(r.dist, r.dist) for r in group]
        vals = [r.avg_reward for r in group]
        errs = [r.reward_std for r in group]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(group))
        ax.bar(x, vals, yerr=errs, capsize=6, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Evaluation avg reward (20 episodes)")
        ax.set_title(f"Experiment 3: performance vs delay variability (mean={m:.3f}s)")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        fname = out_dir / f"exp3_mean_{m:.3f}_reward_bar.png"
        fig.savefig(fname, dpi=180)
        plt.close(fig)


def plot_box_per_mean(rows: list[RunRow], out_dir: Path) -> None:
    means = sorted({r.mean_s for r in rows})
    for m in means:
        group = _sorted_by_regime([r for r in rows if abs(r.mean_s - m) < 1e-9])
        labels = [REGIME_LABEL.get(r.dist, r.dist) for r in group]
        data = [r.episode_rewards for r in group]

        fig, ax = plt.subplots(figsize=(10, 5))
        # Matplotlib 3.9+: "labels" renamed to "tick_labels"
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.set_ylabel("Episode reward (eval episodes)")
        ax.set_title(f"Experiment 3: reward distribution across eval episodes (mean={m:.3f}s)")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        fname = out_dir / f"exp3_mean_{m:.3f}_reward_box.png"
        fig.savefig(fname, dpi=180)
        plt.close(fig)


def plot_degradation(rows: list[RunRow], out_dir: Path) -> None:
    """
    Plot relative performance vs the fixed baseline at the same mean delay.
    This answers: how much does *variability* (not mean) hurt?
    """
    means = sorted({r.mean_s for r in rows})
    fig, ax = plt.subplots(figsize=(10, 5))

    for m in means:
        group = [r for r in rows if abs(r.mean_s - m) < 1e-9]
        fixed = next((r for r in group if r.dist == "fixed"), None)
        if fixed is None:
            continue
        # If the baseline is very small, percent changes explode and become misleading.
        if fixed.avg_reward < 1e-6:
            continue

        xs = []
        ys = []
        for r in _sorted_by_regime(group):
            if r.dist == "fixed":
                continue
            xs.append(REGIME_LABEL.get(r.dist, r.dist))
            ys.append(100.0 * (r.avg_reward / fixed.avg_reward - 1.0))

        ax.plot(xs, ys, marker="o", linewidth=2, label=f"mean={m:.3f}s")

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("Δ reward vs fixed baseline (%)")
    ax.set_title("Experiment 3: performance degradation due to delay variability (relative to fixed)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fname = out_dir / "exp3_degradation_vs_fixed.png"
    fig.savefig(fname, dpi=180)
    plt.close(fig)


def plot_degradation_abs(rows: list[RunRow], out_dir: Path) -> None:
    """
    Absolute delta reward vs the fixed baseline at the same mean delay.
    This is more stable than percent when the fixed baseline is low.
    """
    means = sorted({r.mean_s for r in rows})
    fig, ax = plt.subplots(figsize=(10, 5))

    for m in means:
        group = [r for r in rows if abs(r.mean_s - m) < 1e-9]
        fixed = next((r for r in group if r.dist == "fixed"), None)
        if fixed is None:
            continue

        xs = []
        ys = []
        for r in _sorted_by_regime(group):
            if r.dist == "fixed":
                continue
            xs.append(REGIME_LABEL.get(r.dist, r.dist))
            ys.append(r.avg_reward - fixed.avg_reward)

        ax.plot(xs, ys, marker="o", linewidth=2, label=f"mean={m:.3f}s")

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("Δ reward vs fixed baseline (absolute)")
    ax.set_title("Experiment 3: absolute performance change due to delay variability (vs fixed)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fname = out_dir / "exp3_degradation_vs_fixed_abs.png"
    fig.savefig(fname, dpi=180)
    plt.close(fig)


def plot_reward_vs_delay_tail(rows: list[RunRow], out_dir: Path) -> None:
    """
    Correlate performance with realized delay tail metrics (p90/p99).
    This is the most "insightful" plot for variable delay: it ties heavy tails/spikes to reward.
    """
    usable = [r for r in rows if r.delay_p99_ms is not None]
    if not usable:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for r in usable:
        color = {
            "fixed": "black",
            "uniform": "#1f77b4",
            "normal": "#2ca02c",
            "lognormal": "#ff7f0e",
            "mixture": "#d62728",
        }.get(r.dist, "gray")
        ax.scatter(r.delay_p99_ms, r.avg_reward, color=color, alpha=0.9)
        ax.annotate(f"{r.mean_s:.1f}s", (r.delay_p99_ms, r.avg_reward), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.set_xlabel("Realized delay p99 (ms) during training")
    ax.set_ylabel("Evaluation avg reward")
    ax.set_title("Experiment 3: reward vs delay tail severity (p99)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = out_dir / "exp3_reward_vs_delay_p99.png"
    fig.savefig(fname, dpi=180)
    plt.close(fig)


def plot_reward_vs_delay_p50(rows: list[RunRow], out_dir: Path) -> None:
    """
    Same idea as p99, but using p50 (median) delay.
    Useful because mixture distributions can have a large p99 but a small typical delay.
    """
    usable = [r for r in rows if r.delay_p50_ms is not None]
    if not usable:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for r in usable:
        color = {
            "fixed": "black",
            "uniform": "#1f77b4",
            "normal": "#2ca02c",
            "lognormal": "#ff7f0e",
            "mixture": "#d62728",
        }.get(r.dist, "gray")
        ax.scatter(r.delay_p50_ms, r.avg_reward, color=color, alpha=0.9)
        ax.annotate(f"{r.mean_s:.1f}s", (r.delay_p50_ms, r.avg_reward), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.set_xlabel("Realized delay p50 (ms) during training")
    ax.set_ylabel("Evaluation avg reward")
    ax.set_title("Experiment 3: reward vs typical delay (median/p50)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = out_dir / "exp3_reward_vs_delay_p50.png"
    fig.savefig(fname, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Experiment 3 variable-delay results")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Path to ONE results/comparison_results.json")
    parser.add_argument("--results_files", type=str, nargs="+", default=None,
                        help="Paths to MULTIPLE comparison_results.json files (for cross-mean plots)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Directory to save plots (default: same dir as results_file)")
    args = parser.parse_args()

    if not args.results_file and not args.results_files:
        raise SystemExit("Provide either --results_file or --results_files")
    if args.results_file and args.results_files:
        raise SystemExit("Provide only one of --results_file or --results_files")

    if args.results_file:
        results_file = Path(args.results_file).resolve()
        out_dir = Path(args.out_dir).resolve() if args.out_dir else results_file.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = load_runs(results_file)

        plot_bars_per_mean(rows, out_dir)
        plot_box_per_mean(rows, out_dir)
        plot_degradation(rows, out_dir)
        plot_reward_vs_delay_tail(rows, out_dir)
        plot_degradation_abs(rows, out_dir)
        plot_reward_vs_delay_p50(rows, out_dir)
        # cross-mean style plots (single mean will still work, just 1 point/line)
        plot_reward_vs_mean_by_regime(rows, out_dir)
        plot_degradation_vs_fixed_across_means(rows, out_dir)
        plot_abs_delta_vs_fixed_across_means(rows, out_dir)

        print(f"Saved plots to: {out_dir}")
        return

    results_files = [Path(p).resolve() for p in args.results_files]
    out_dir = Path(args.out_dir).resolve() if args.out_dir else results_files[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_runs_multi(results_files)
    plot_reward_vs_mean_by_regime(rows, out_dir)
    plot_degradation_vs_fixed_across_means(rows, out_dir)
    plot_abs_delta_vs_fixed_across_means(rows, out_dir)
    plot_reward_vs_delay_tail(rows, out_dir)
    plot_reward_vs_delay_p50(rows, out_dir)

    print(f"Saved combined plots to: {out_dir}")


if __name__ == "__main__":
    main()


