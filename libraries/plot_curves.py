"""Plot mean +- std learning curves across seeds for one or more algorithm runs.

Execution:
    python -m libraries.plot_curves --runs dqn:base reinforce:base --out outputs/figures/learning_curves.png
    python -m libraries.plot_curves --runs dqn:base --source eval --out outputs/figures/dqn_eval_curve.png
    python -m libraries.plot_curves --runs dqn:base dqn:nodouble --out outputs/figures/ablation_double.png
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import DEFAULT_LOG_DIR, ensure_dir, trailing_ma

DEFAULT_RESULTS_DIR = "outputs/results"
DEFAULT_SOURCE = "train"
DEFAULT_XAXIS = "step"
DEFAULT_WINDOW = 20
DEFAULT_SOLVED_THRESHOLD = 475


def load_group(algo: str, tag: str, source: str = DEFAULT_SOURCE,
               xaxis: str = DEFAULT_XAXIS, window: int = DEFAULT_WINDOW):
    """Load all CSV logs for a given (algo, tag) run group and extract (x, y) series.

    Input:  algo   -- algorithm name ('dqn', 'reinforce');
            tag    -- run tag identifier ('base', 'nodouble', etc.);
            source -- log type, either 'train' (episode log) or 'eval' (evaluation log);
            xaxis  -- x-axis metric ('step' or 'episode', applied when source='train');
            window -- trailing moving average window size for training logs.
    Output: list of (x, y) tuples per seed, where x and y are float ndarrays.
    Flow:   glob matching CSV log files, parse DataFrames, select columns by source/xaxis,
            apply trailing_ma when source='train', collect and return series list.
    """
    series = []
    suffix = "_eval.csv" if source == "eval" else ".csv"
    for f in sorted(glob.glob(f"{DEFAULT_LOG_DIR}/{algo}/{tag}_seed*{suffix}")):
        if source != "eval" and f.endswith("_eval.csv"):
            continue
        d = pd.read_csv(f)
        if len(d) == 0:
            continue
        if source == "eval":
            x, y = d["env_step"].to_numpy(float), d["eval_mean"].to_numpy(float)
        else:
            x = d["env_step"].to_numpy(float) if xaxis == "step" else d["episode"].to_numpy(float)
            y = trailing_ma(d["return"].to_numpy(float), window)
        series.append((x, y))
    return series


def main():
    """CLI entrypoint: generate aggregate learning curve plots and save PNG figure.

    Input:  --runs (list of 'algo:tag'), --source, --xaxis, --window, --out.
    Output: PNG figure saved to --out.
    Flow:   parse arguments, call load_group for each run, interpolate across a common grid,
            compute mean and standard deviation, plot shaded ribbons, add baseline reference, save figure.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--source", choices=["train", "eval"], default=DEFAULT_SOURCE)
    ap.add_argument("--xaxis", choices=["step", "episode"], default=DEFAULT_XAXIS)
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.out))
    plt.figure(figsize=(8, 5))
    for run in args.runs:
        algo, tag = run.split(":")
        series = load_group(algo, tag, args.source, args.xaxis, args.window)
        if not series:
            print(f"WARNING: no log found for {run}")
            continue
        xmax = min(s[0][-1] for s in series)
        grid = np.linspace(0, xmax, 300)
        ys = np.stack([np.interp(grid, x, y) for x, y in series])
        mu, sd = ys.mean(0), ys.std(0)
        plt.plot(grid, mu, label=f"{algo}:{tag} (n={len(series)} seeds)")
        plt.fill_between(grid, mu - sd, mu + sd, alpha=0.2)

    bl_path = f"{DEFAULT_RESULTS_DIR}/baseline_random.csv" if os.path.exists(f"{DEFAULT_RESULTS_DIR}/baseline_random.csv") else "results/baseline_random.csv"
    if os.path.exists(bl_path):
        rnd = pd.read_csv(bl_path)["return"].mean()
        plt.axhline(rnd, linestyle="--", color="gray", label=f"random ({rnd:.0f})")
    plt.axhline(DEFAULT_SOLVED_THRESHOLD, linestyle=":", color="green", label="solved threshold (475)")
    plt.xlabel("Environment steps" if args.xaxis == "step" or args.source == "eval" else "Episode")
    plt.ylabel("Return (eval, eps=0)" if args.source == "eval" else f"Return (MA{args.window})")
    plt.title("Learning curves (mean +- std across seeds)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print("Saved", args.out)


if __name__ == "__main__":
    main()

