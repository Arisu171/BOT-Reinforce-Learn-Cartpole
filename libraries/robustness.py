"""Observation noise robustness evaluation: test V0-trained policies on V2 across noise levels.

Execution:
    python -m libraries.robustness --runs dqn:base reinforce:base --sigmas 0 0.01 0.05 0.1

Outputs:
    outputs/results/noise_robustness.csv
    outputs/figures/noise_robustness.png
"""
import argparse
import glob
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .evaluate import evaluate_policy
from .policies import load_policy
from .utils import TEST_SEED_START, ensure_dir

DEFAULT_RESULTS_DIR = "outputs/results"
DEFAULT_FIGURES_DIR = "outputs/figures"
DEFAULT_OUT_CSV = f"{DEFAULT_RESULTS_DIR}/noise_robustness.csv"
DEFAULT_OUT_FIG = f"{DEFAULT_FIGURES_DIR}/noise_robustness.png"
DEFAULT_SIGMAS = [0.0, 0.01, 0.05, 0.1]
DEFAULT_EPISODES = 50
DEFAULT_WHICH = "best"


def main():
    """CLI entrypoint: evaluate policy robustness across observation noise levels.

    Input:  --runs (list of 'algo:tag'), --sigmas (list of float), --episodes (int),
            --which ('best'|'final'), --out_csv, --out_fig.
    Output: outputs/results/noise_robustness.csv and outputs/figures/noise_robustness.png.
    Flow:   iterate through run/seed/sigma combinations, load policies, evaluate with evaluate_policy,
            aggregate results into DataFrame, plot errorbar chart and save figure.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--sigmas", nargs="+", type=float, default=DEFAULT_SIGMAS)
    ap.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    ap.add_argument("--which", choices=["best", "final"], default=DEFAULT_WHICH)
    ap.add_argument("--out_csv", default=DEFAULT_OUT_CSV)
    ap.add_argument("--out_fig", default=DEFAULT_OUT_FIG)
    args = ap.parse_args()

    ensure_dir(DEFAULT_RESULTS_DIR)
    ensure_dir(DEFAULT_FIGURES_DIR)
    rows = []
    for run in args.runs:
        algo, tag = run.split(":")
        for path in sorted(glob.glob(f"models/{algo}_{tag}_seed*_{args.which}.pt")):
            seed = int(re.search(r"_seed(\d+)_", path).group(1))
            act = load_policy(algo, path)
            for sg in args.sigmas:
                variant = "v0" if sg == 0 else "v2"
                res = evaluate_policy(act, n_episodes=args.episodes, seed_start=TEST_SEED_START,
                                      variant=variant, sigma=sg)
                rows.append({"algo": algo, "tag": tag, "seed": seed, "sigma": sg, "mean": res["mean"]})
                print(f"{run} seed={seed} sigma={sg}: {res['mean']:.1f}")
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    plt.figure(figsize=(7, 4.5))
    for (algo, tag), g in df.groupby(["algo", "tag"]):
        agg = g.groupby("sigma")["mean"].agg(["mean", "std"]).fillna(0)
        plt.errorbar(agg.index, agg["mean"], yerr=agg["std"], marker="o", capsize=3, label=f"{algo}:{tag}")
    plt.xlabel("Observation noise sigma")
    plt.ylabel("Mean return (test seeds)")
    plt.title("Noise robustness (trained on V0)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=150)
    print("Saved", args.out_csv, "and", args.out_fig)


if __name__ == "__main__":
    main()

