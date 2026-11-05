"""Evaluate non-learning baseline policies (random or heuristic) and save CSV and histogram PNG artifacts.

Run:  python -m libraries.baseline --policy random
Save: outputs/results/baseline_<policy>.csv  (5 policy_seed x 100 episodes on test seeds 100..199)
      outputs/figures/<policy>_hist.png
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import evaluate_policy
from .policies import heuristic_policy, random_policy
from .utils import TEST_SEED_START, TRAIN_SEEDS, ensure_dir

DEFAULT_RESULTS_DIR = "outputs/results"
DEFAULT_FIGURES_DIR = "outputs/figures"
DEFAULT_POLICY = "random"
DEFAULT_EPISODES = 100
DEFAULT_BINS = 30


def run_baseline(policy_name: str = DEFAULT_POLICY, episodes: int = DEFAULT_EPISODES,
                 out_csv: str | None = None, out_png: str | None = None) -> dict:
    """Execute baseline evaluation and export return distribution statistics.

    Input:  policy_name -- "random" or "heuristic" (default DEFAULT_POLICY);
            episodes    -- number of episodes per policy seed (default DEFAULT_EPISODES = 100);
            out_csv     -- output CSV path (default None, generated from policy name);
            out_png     -- output histogram PNG path (default None, generated from policy name).
    Output: dict containing statistics: mean, std, min, max, median, and returns list.
    Flow:   iterate through TRAIN_SEEDS, instantiate policy, evaluate via evaluate_policy,
            save results to CSV, plot return histogram to PNG, return metrics dictionary.
    """
    ensure_dir(DEFAULT_RESULTS_DIR)
    ensure_dir(DEFAULT_FIGURES_DIR)
    csv_path = out_csv or f"{DEFAULT_RESULTS_DIR}/baseline_{policy_name}.csv"
    png_path = out_png or f"{DEFAULT_FIGURES_DIR}/{policy_name}_hist.png"

    rows = []
    for ps in TRAIN_SEEDS:
        pol = random_policy(ps) if policy_name == "random" else heuristic_policy()
        res = evaluate_policy(pol, n_episodes=episodes, seed_start=TEST_SEED_START)
        for i, ret in enumerate(res["returns"]):
            rows.append({
                "policy": policy_name,
                "policy_seed": ps,
                "episode": i,
                "env_seed": TEST_SEED_START + i,
                "return": ret
            })
        if policy_name == "heuristic":
            break

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rets = np.array([r["return"] for r in rows])
    stats = {
        "n": len(rets),
        "mean": float(rets.mean()),
        "std": float(rets.std()),
        "min": float(rets.min()),
        "max": float(rets.max()),
        "median": float(np.median(rets)),
        "returns": [float(x) for x in rets],
    }

    plt.figure(figsize=(6, 4))
    plt.hist(rets, bins=DEFAULT_BINS)
    plt.xlabel("Episode length (return)")
    plt.ylabel("Episode count")
    plt.title(f"Return distribution: {policy_name} policy")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    return stats


def main():
    """CLI entry point for baseline evaluation.

    Input:  None.
    Output: None (prints summary statistics to stdout).
    Flow:   parse CLI arguments, invoke run_baseline().
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["random", "heuristic"], default=DEFAULT_POLICY)
    ap.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    args = ap.parse_args()

    stats = run_baseline(policy_name=args.policy, episodes=args.episodes)
    print(f"[{args.policy}] n={stats['n']} mean={stats['mean']:.2f} std={stats['std']:.2f} "
          f"min={stats['min']:.0f} max={stats['max']:.0f} median={stats['median']:.1f}")
    print(f"Saved {DEFAULT_RESULTS_DIR}/baseline_{args.policy}.csv and {DEFAULT_FIGURES_DIR}/{args.policy}_hist.png")


if __name__ == "__main__":
    main()
