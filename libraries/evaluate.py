"""Evaluate a policy across n_episodes independent episodes and collect raw unshaped returns.
Never modify this file to artificially inflate metrics.
"""
import numpy as np

from .envs import DEFAULT_MASSPOLE, DEFAULT_SIGMA, make_env
from .utils import TEST_SEED_START

DEFAULT_EVAL_EPISODES = 100
DEFAULT_VARIANT = "v0"


def evaluate_policy(select_action, n_episodes: int = DEFAULT_EVAL_EPISODES,
                    seed_start: int = TEST_SEED_START,
                    variant: str = DEFAULT_VARIANT, sigma: float = DEFAULT_SIGMA,
                    masspole: float = DEFAULT_MASSPOLE) -> dict:
    """Run n_episodes evaluation episodes and collect raw return metrics.

    Input:  select_action -- callable mapping obs -> action integer (0 or 1);
            n_episodes    -- number of evaluation episodes (default DEFAULT_EVAL_EPISODES = 100);
            seed_start    -- initial environment evaluation seed (default TEST_SEED_START = 100);
            variant       -- environment variant string (default DEFAULT_VARIANT = "v0");
            sigma         -- noise standard deviation for variant "v2" (default DEFAULT_SIGMA);
            masspole      -- mass of the pole in kg (default DEFAULT_MASSPOLE = 0.1 kg).
    Output: dict with keys mean, std, min, max, frac_500, and returns (list of floats).
    Flow:   for each episode with seed = seed_start + i, instantiate env, run until termination,
            accumulate info.get("raw_reward", r), close env, compute and return summary stats.
    """
    rets = []
    for i in range(n_episodes):
        sd = seed_start + i
        env = make_env(variant, seed=sd, sigma=sigma, masspole=masspole)
        obs, _ = env.reset(seed=sd)
        done, total = False, 0.0
        while not done:
            obs, r, terminated, truncated, info = env.step(int(select_action(obs)))
            total += info.get("raw_reward", r)
            done = terminated or truncated
        env.close()
        rets.append(total)
    arr = np.asarray(rets, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "frac_500": float((arr >= 500).mean()),
        "returns": [float(x) for x in arr],
    }
