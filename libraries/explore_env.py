"""CartPole-v1 environment exploration: print space specifications, sample random steps, and plot observation histograms.

Execution:
    python -m libraries.explore_env

Outputs:
    outputs/figures/obs_hist.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .envs import make_env
from .utils import ensure_dir

DEFAULT_STEPS = 1000
DEFAULT_SEED = 0
DEFAULT_VARIANT = "v0"
DEFAULT_OUT_FIG = "outputs/figures/obs_hist.png"


def main():
    """CLI entrypoint: explore environment state spaces and save observation distributions.

    Input:  None.
    Output: prints observation_space, action_space, max_episode_steps, and variable statistics;
            saves outputs/figures/obs_hist.png.
    Flow:   instantiate V0 environment, collect random transitions for DEFAULT_STEPS with DEFAULT_SEED,
            compute min/max/mean for each dimension, plot 4-subplot histogram, save figure.
    """
    env = make_env(DEFAULT_VARIANT)
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)
    print("max_episode_steps:", getattr(getattr(env, "spec", None), "max_episode_steps", "?"))
    rng = np.random.default_rng(DEFAULT_SEED)
    obs, _ = env.reset(seed=DEFAULT_SEED)
    data = [obs]
    for _ in range(DEFAULT_STEPS):
        obs, r, te, tr, _ = env.step(int(rng.integers(0, 2)))
        data.append(obs)
        if te or tr:
            obs, _ = env.reset()
    data = np.array(data)
    names = ["x (cart position)", "x_dot (cart velocity)", "theta (pole angle)", "theta_dot (angular velocity)"]
    for i, n in enumerate(names):
        print(f"{n}: min={data[:, i].min():.3f} max={data[:, i].max():.3f} mean={data[:, i].mean():.3f}")
    ensure_dir("outputs/figures")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    for i, ax in enumerate(axes):
        ax.hist(data[:, i], bins=30)
        ax.set_title(names[i], fontsize=9)
    plt.tight_layout()
    plt.savefig(DEFAULT_OUT_FIG, dpi=150)
    print(f"Saved {DEFAULT_OUT_FIG}")


if __name__ == "__main__":
    main()

