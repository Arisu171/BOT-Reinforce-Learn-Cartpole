"""Failure mode and policy analysis: classify termination reasons, plot trajectories and policy heatmap.

Execution:
    python -m libraries.error_analysis --algo dqn --model models/dqn_base_seed0_best.pt

Outputs:
    outputs/results/error_analysis.csv
    outputs/results/error_analysis_summary.json
    outputs/figures/trajectories.png
    outputs/figures/policy_heatmap.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .envs import THETA_LIMIT, make_env
from .policies import get_policy
from .utils import ANALYSIS_SEED_START, ensure_dir

DEFAULT_RESULTS_DIR = "outputs/results"
DEFAULT_FIGURES_DIR = "outputs/figures"
DEFAULT_EPISODES = 500
DEFAULT_VARIANT = "v0"
DEFAULT_SIGMA = 0.05
DEFAULT_OUT_CSV = f"{DEFAULT_RESULTS_DIR}/error_analysis.csv"
DEFAULT_OUT_SUMMARY_JSON = f"{DEFAULT_RESULTS_DIR}/error_analysis_summary.json"
DEFAULT_TRAJECTORIES_PNG = f"{DEFAULT_FIGURES_DIR}/trajectories.png"
DEFAULT_POLICY_HEATMAP_PNG = f"{DEFAULT_FIGURES_DIR}/policy_heatmap.png"


def run_episode(act, seed: int, variant: str = DEFAULT_VARIANT, sigma: float = DEFAULT_SIGMA):
    """Execute a single evaluation episode and record state trajectory.

    Input:  act     -- policy action selector callable;
            seed    -- environment random seed;
            variant -- environment variant identifier (default DEFAULT_VARIANT);
            sigma   -- noise standard deviation for observation noise (default DEFAULT_SIGMA).
    Output: tuple (row, xs, ths) containing episode metrics dictionary,
            cart position trajectory, and pole angle trajectory.
    Flow:   instantiate environment, reset with seed, step until termination,
            classify termination reason into timeout, pole_angle, or cart_position.
    """
    env = make_env(variant, seed=seed, sigma=sigma)
    obs, _ = env.reset(seed=seed)
    xs, ths = [float(env.unwrapped.state[0])], [float(env.unwrapped.state[2])]
    done, steps, truncated = False, 0, False
    while not done:
        obs, r, terminated, truncated, info = env.step(int(act(obs)))
        st = env.unwrapped.state
        xs.append(float(st[0]))
        ths.append(float(st[2]))
        steps += 1
        done = terminated or truncated
    if truncated:
        reason = "timeout"
    elif abs(ths[-1]) > THETA_LIMIT:
        reason = "pole_angle"
    else:
        reason = "cart_position"
    env.close()
    return {"seed": seed, "length": steps, "reason": reason,
            "final_x": xs[-1], "final_theta": ths[-1]}, xs, ths


def main():
    """CLI entrypoint: execute error analysis across evaluation episodes and save artifacts.

    Input:  --algo, --model, --episodes, --variant, --sigma.
    Output: outputs/results/error_analysis.csv, outputs/results/error_analysis_summary.json,
            outputs/figures/trajectories.png, outputs/figures/policy_heatmap.png.
    Flow:   run episodes, log failure causes, identify best and worst trajectories,
            generate trajectory plots and state-action heatmap, persist outputs.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["dqn", "reinforce", "random", "heuristic"], required=True)
    ap.add_argument("--model")
    ap.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    ap.add_argument("--variant", default=DEFAULT_VARIANT)
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    args = ap.parse_args()

    ensure_dir(DEFAULT_RESULTS_DIR)
    ensure_dir(DEFAULT_FIGURES_DIR)
    act = get_policy(args.algo, args.model)
    rows, traj = [], {}
    for i in range(args.episodes):
        row, xs, ths = run_episode(act, ANALYSIS_SEED_START + i, args.variant, args.sigma)
        rows.append(row)
        traj[row["seed"]] = (xs, ths)
    df = pd.DataFrame(rows)
    df.to_csv(DEFAULT_OUT_CSV, index=False)

    summary = {"n": len(df), "mean_length": float(df["length"].mean()),
               "reason_counts": df["reason"].value_counts().to_dict(),
               "n_below_200": int((df["length"] < 200).sum()),
               "worst_seeds": df.nsmallest(5, "length")["seed"].tolist()}
    with open(DEFAULT_OUT_SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    best = df.loc[df["length"].idxmax()]
    worst = df.loc[df["length"].idxmin()]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
    for lab, row in (("best", best), ("worst", worst)):
        xs, ths = traj[int(row["seed"])]
        axes[0].plot(xs, label=f"{lab} (len={int(row['length'])}, {row['reason']})")
        axes[1].plot(ths, label=lab)
    axes[0].axhline(2.4, ls="--", c="r")
    axes[0].axhline(-2.4, ls="--", c="r")
    axes[0].set_ylabel("x (m)")
    axes[0].legend()
    axes[1].axhline(THETA_LIMIT, ls="--", c="r")
    axes[1].axhline(-THETA_LIMIT, ls="--", c="r")
    axes[1].set_ylabel("theta (rad)")
    axes[1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(DEFAULT_TRAJECTORIES_PNG, dpi=150)
    plt.close()

    ths = np.linspace(-THETA_LIMIT, THETA_LIMIT, 41)
    dths = np.linspace(-2.0, 2.0, 41)
    grid = np.array([[act(np.array([0.0, 0.0, th, dth], dtype=np.float32))
                      for th in ths] for dth in dths])
    plt.figure(figsize=(6, 5))
    plt.imshow(grid, origin="lower", aspect="auto",
               extent=[ths[0], ths[-1], dths[0], dths[-1]], cmap="coolwarm", vmin=0, vmax=1)
    plt.colorbar(ticks=[0, 1], label="action (0=left, 1=right)")
    plt.xlabel("theta (rad)")
    plt.ylabel("theta_dot (rad/s)")
    plt.title("Policy slice at x=0, x_dot=0")
    plt.tight_layout()
    plt.savefig(DEFAULT_POLICY_HEATMAP_PNG, dpi=150)
    print(f"Saved {DEFAULT_OUT_CSV}, {DEFAULT_TRAJECTORIES_PNG}, {DEFAULT_POLICY_HEATMAP_PNG}")


if __name__ == "__main__":
    main()

