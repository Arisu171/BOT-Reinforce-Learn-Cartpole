"""Generate notebooks/04_demo.ipynb as pure JSON without external dependencies.

Execution:
    python scripts/make_demo_notebook.py --algo dqn --model models/dqn_base_seed0_best.pt
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from libraries.envs import DEFAULT_MASSPOLE, DEFAULT_SIGMA
from libraries.utils import ensure_dir

DEFAULT_ALGO = "dqn"
DEFAULT_OUT = "notebooks/04_demo.ipynb"
DEFAULT_DEMO_SEED = 4242
DEFAULT_VARIANT = "v0"


def md(t):
    """Create a Markdown cell dictionary for Jupyter Notebook.

    Input:  t -- Markdown string content.
    Output: dictionary representing a standard nbformat 4 markdown cell.
    Flow:   splitlines preserving line endings and wrap into cell structure.
    """
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(True)}


def code(t):
    """Create a Code cell dictionary for Jupyter Notebook.

    Input:  t -- Python code string content.
    Output: dictionary representing an unexecuted nbformat 4 code cell.
    Flow:   splitlines preserving line endings and construct cell structure with empty outputs.
    """
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(True)}


def main():
    """CLI entrypoint: generate notebooks/04_demo.ipynb.

    Input:  --algo (default DEFAULT_ALGO), --model (required), --out (default DEFAULT_OUT).
    Output: complete runnable .ipynb JSON notebook file.
    Flow:   parse arguments, build list of markdown and code cells, serialize to JSON.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default=DEFAULT_ALGO)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()

    cells = [
        md("# 04. Demo: Balancing CartPole-v1 Agent\n\n"
           "This notebook interactively evaluates the balance control of a trained reinforcement learning agent:\n"
           "- Load best policy checkpoint from `models/`.\n"
           "- Test on an unseen random seed.\n"
           "- Compare performance and animated simulation against the Random baseline policy."),
        code("import os\n"
             "import sys\n"
             "sys.path.insert(0, os.path.abspath('..'))\n"
             "os.chdir('..')\n\n"
             "import numpy as np\n"
             "import matplotlib.pyplot as plt\n"
             "from IPython.display import Image, display\n"
             "import libraries.envs as env_module\n"
             "from libraries.envs import make_env, DEFAULT_MASSPOLE, DEFAULT_SIGMA\n"
             "from libraries.policies import get_policy\n\n"
             f"ALGO = '{a.algo}'\n"
             f"MODEL = '{a.model}'\n"
             "act = get_policy(ALGO, MODEL)\n"
             "rnd = get_policy('random', seed=0)"),
        md("## 1. Environment and Hyperparameter Setup\n\n"
           "Default parameters are imported from `libraries.envs` (`DEFAULT_MASSPOLE`, `DEFAULT_SIGMA`).\n"
           "Change `SEED` to test different initial angles.\n"
           "To evaluate robustness under observation noise, set `VARIANT = 'v2'` and adjust `SIGMA`.\n"
           "To test heavier pole dynamics (e.g. 0.3 kg or 0.5 kg), modify `MASSPOLE` below or set `env_module.DEFAULT_MASSPOLE = 0.5`."),
        code(f"SEED = {DEFAULT_DEMO_SEED}\n"
             f"VARIANT = '{DEFAULT_VARIANT}'\n"
             "SIGMA = DEFAULT_SIGMA\n"
             "MASSPOLE = DEFAULT_MASSPOLE\n\n"
             "def run(policy, seed, variant=VARIANT, sigma=SIGMA, masspole=MASSPOLE):\n"
             "    env = make_env(variant, seed=seed, sigma=sigma, render_mode='rgb_array', masspole=masspole)\n"
             "    obs, _ = env.reset(seed=seed)\n"
             "    frames = []\n"
             "    ret = 0.0\n"
             "    done = False\n"
             "    while not done and len(frames) < 500:\n"
             "        frames.append(env.render())\n"
             "        obs, r, te, tr, info = env.step(int(policy(obs)))\n"
             "        ret += info.get('raw_reward', r)\n"
             "        done = te or tr\n"
             "    env.close()\n"
             "    return frames, ret\n\n"
             "f_rnd, r_rnd = run(rnd, SEED)\n"
             "f_agt, r_agt = run(act, SEED)\n"
             "print(f'Random policy return: {r_rnd:.0f} | Trained agent return: {r_agt:.0f}')"),
        md("## 2. Render and Display Animated Simulations\n\n"
           "The two GIF animations below illustrate balancing behavior for the random policy and the trained agent."),
        code("import imageio\n\n"
             "os.makedirs('outputs/videos', exist_ok=True)\n"
             "imageio.mimsave('outputs/videos/_demo_random.gif', f_rnd[::2], fps=25, loop=0)\n"
             "imageio.mimsave('outputs/videos/_demo_agent.gif', f_agt[::2], fps=25, loop=0)\n\n"
             "display(Image(filename='outputs/videos/_demo_random.gif'))\n"
             "display(Image(filename='outputs/videos/_demo_agent.gif'))"),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    ensure_dir(os.path.dirname(a.out))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Saved", a.out)


if __name__ == "__main__":
    main()

