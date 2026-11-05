"""Record an evaluation episode of any policy to an animated GIF.

Execution:
    python -m libraries.record_gif --algo random --out outputs/videos/random.gif --seed 3000
    python -m libraries.record_gif --algo dqn --model models/dqn_base_seed0_best.pt --out outputs/videos/dqn_best.gif --seed 3000
    python -m libraries.record_gif --algo dqn --model ... --variant v2 --sigma 0.05 --out outputs/videos/dqn_noisy.gif
"""
import argparse
import os

import imageio

from .envs import DEFAULT_MASSPOLE, DEFAULT_SIGMA, make_env
from .policies import get_policy
from .utils import DEMO_SEED_START, ensure_dir

DEFAULT_FPS = 25
DEFAULT_STRIDE = 2
DEFAULT_MAX_FRAMES = 500
DEFAULT_SEED = DEMO_SEED_START
DEFAULT_VARIANT = "v0"


def save_gif(frames, path: str, fps: int = DEFAULT_FPS) -> None:
    """Save a sequence of RGB image frames as an animated GIF file.

    Input:  frames -- list of ndarray uint8 image frames of shape (H, W, 3);
            path   -- target output file path ending in .gif;
            fps    -- playback frames per second (default DEFAULT_FPS = 25).
    Output: None (writes GIF file to disk).
    Flow:   attempt imageio.mimsave with fps and loop=0; fall back to duration parameter on TypeError.
    """
    try:
        imageio.mimsave(path, frames, fps=fps, loop=0)
    except (TypeError, ValueError):
        imageio.mimsave(path, frames, duration=1000.0 / fps, loop=0)


def main():
    """CLI entrypoint: record episode frames and save animated GIF.

    Input:  --algo, --model, --out, --seed, --variant, --sigma, --max_frames, --stride, --masspole.
    Output: animated GIF saved at --out; prints step count and episode return.
    Flow:   parse CLI options, load policy, create environment with rgb_array render mode,
            step environment while capturing frames every stride steps, invoke save_gif.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["random", "heuristic", "dqn", "reinforce"], required=True)
    ap.add_argument("--model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--variant", default=DEFAULT_VARIANT)
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    ap.add_argument("--masspole", type=float, default=DEFAULT_MASSPOLE)
    ap.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES)
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.out))
    act = get_policy(args.algo, args.model, seed=args.seed)
    env = make_env(args.variant, seed=args.seed, sigma=args.sigma,
                   masspole=args.masspole, render_mode="rgb_array")
    obs, _ = env.reset(seed=args.seed)
    frames, steps, ret, done = [], 0, 0.0, False
    while not done and steps < args.max_frames:
        if steps % args.stride == 0:
            frames.append(env.render())
        obs, r, te, tr, info = env.step(int(act(obs)))
        ret += info.get("raw_reward", r)
        steps += 1
        done = te or tr
    env.close()

    save_gif(frames, args.out, fps=args.fps)
    print(f"Saved {args.out}: {steps} steps, return={int(ret)}, seed={args.seed}, masspole={args.masspole}kg")


if __name__ == "__main__":
    main()
