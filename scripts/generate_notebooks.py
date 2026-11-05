"""Generate Jupyter Notebook files for the CartPole RL project.

Generates:
  notebooks/01_explore_env.ipynb
  notebooks/02_train_dqn.ipynb
  notebooks/03_train_reinforce.ipynb
  notebooks/05_analysis.ipynb
"""
import json
import os


def make_md_cell(text: str) -> dict:
    """Create a Markdown cell dictionary for Jupyter Notebook.

    Input:  text -- markdown string content.
    Output: dictionary representing a standard nbformat 4 markdown cell.
    Flow:   splitlines and package into cell dictionary.
    """
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(True)
    }


def make_code_cell(code: str) -> dict:
    """Create a Code cell dictionary for Jupyter Notebook.

    Input:  code -- Python code string content.
    Output: dictionary representing an unexecuted nbformat 4 code cell.
    Flow:   splitlines and package into cell dictionary with empty outputs.
    """
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(True)
    }


def save_notebook(cells: list, path: str) -> None:
    """Save cell list to a .ipynb file formatted as valid JSON.

    Input:  cells -- list of notebook cells (markdown or code);
            path  -- destination .ipynb file path.
    Output: None (writes file to disk).
    Flow:   build standard nbformat 4 dictionary and write JSON with indent 1.
    """
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)


def generate_01_explore() -> None:
    """Generate notebooks/01_explore_env.ipynb to explore the CartPole environment.

    Input:  None.
    Output: notebooks/01_explore_env.ipynb file.
    Flow:   construct markdown and code cells, invoke save_notebook.
    """
    cells = [
        make_md_cell(
            "# 01. CartPole-v1 Environment Exploration and Baseline Evaluation\n\n"
            "This notebook guides you through:\n"
            "1. Exploring the observation (state) space and action space.\n"
            "2. Executing a sample episode with the Random policy.\n"
            "3. Evaluating the Random Baseline over 500 episodes and visualizing the return distribution."
        ),
        make_code_cell(
            "import os\n"
            "import sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "os.chdir('..')\n\n"
            "import gymnasium as gym\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from libraries.envs import make_env\n"
            "from libraries.policies import get_policy\n"
            "from libraries.evaluate import evaluate_policy"
        ),
        make_md_cell(
            "## 1. State and Action Spaces\n\n"
            "CartPole-v1 observation vector consists of 4 continuous variables:\n"
            "- $x$: Cart position\n"
            "- $\\dot{x}$: Cart velocity\n"
            "- $\\theta$: Pole angle\n"
            "- $\\dot{\\theta}$: Pole angular velocity\n\n"
            "Actions are discrete: `0` (push cart left) or `1` (push cart right)."
        ),
        make_code_cell(
            "env = make_env('v0', seed=42)\n"
            "obs, info = env.reset(seed=42)\n"
            "print('Observation space:', env.observation_space)\n"
            "print('Action space:', env.action_space)\n"
            "print('Initial observation:', obs)\n"
            "env.close()"
        ),
        make_md_cell(
            "## 2. Test Single Episode with Random Policy"
        ),
        make_code_cell(
            "env = make_env('v0', seed=42)\n"
            "obs, info = env.reset(seed=42)\n"
            "total_reward = 0.0\n"
            "steps = 0\n"
            "done = False\n\n"
            "while not done:\n"
            "    action = env.action_space.sample()\n"
            "    obs, reward, terminated, truncated, info = env.step(action)\n"
            "    total_reward += reward\n"
            "    steps += 1\n"
            "    done = terminated or truncated\n\n"
            "print(f'Episode finished after {steps} steps. Total return = {total_reward}')\n"
            "env.close()"
        ),
        make_md_cell(
            "## 3. Evaluate Random Baseline (500 episodes)\n\n"
            "Theoretically, a random policy on CartPole-v1 achieves an average return around 20-25 points."
        ),
        make_code_cell(
            "policy_random = get_policy('random', seed=123)\n"
            "eval_res = evaluate_policy(policy_random, n_episodes=500, seed_start=100)\n"
            "print(f'Mean return: {eval_res[\"mean\"]:.2f}')\n"
            "print(f'Std:         {eval_res[\"std\"]:.2f}')\n"
            "print(f'Min / Max:   {eval_res[\"min\"]:.1f} / {eval_res[\"max\"]:.1f}')"
        ),
        make_md_cell(
            "## 4. Episode Length Distribution of Random Baseline"
        ),
        make_code_cell(
            "returns = eval_res['returns']\n"
            "plt.figure(figsize=(8, 4.5))\n"
            "plt.hist(returns, bins=25, color='steelblue', edgecolor='black', alpha=0.8)\n"
            "plt.axvline(eval_res['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean = {eval_res[\"mean\"]:.1f}')\n"
            "plt.title('Return Distribution of Random Policy (500 episodes)')\n"
            "plt.xlabel('Return')\n"
            "plt.ylabel('Episode count')\n"
            "plt.grid(True, alpha=0.3)\n"
            "plt.legend()\n"
            "plt.show()"
        )
    ]
    save_notebook(cells, "notebooks/01_explore_env.ipynb")


def generate_02_train_dqn() -> None:
    """Generate notebooks/02_train_dqn.ipynb to demonstrate Double DQN training.

    Input:  None.
    Output: notebooks/02_train_dqn.ipynb file.
    Flow:   construct markdown and code cells, invoke save_notebook.
    """
    cells = [
        make_md_cell(
            "# 02. Double DQN Training on CartPole-v1\n\n"
            "This notebook guides you through:\n"
            "1. Loading standard hyperparameters from `config/dqn.yaml`.\n"
            "2. Training Double DQN with Experience Replay and Soft Target Updates.\n"
            "3. Visualizing training learning curves and exploration schedules.\n"
            "4. Evaluating the best checkpoint on independent test seeds (100-199)."
        ),
        make_code_cell(
            "import os\n"
            "import sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "os.chdir('..')\n\n"
            "import argparse\n"
            "import yaml\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from libraries.dqn import train, DQNAgent\n"
            "from libraries.policies import get_policy\n"
            "from libraries.evaluate import evaluate_policy"
        ),
        make_md_cell(
            "## 1. Load Double DQN Hyperparameters"
        ),
        make_code_cell(
            "with open('config/dqn.yaml', 'r') as f:\n"
            "    cfg = yaml.safe_load(f)\n"
            "print('Default configuration:')\n"
            "for k, v in cfg.items():\n"
            "    print(f'  {k:20s}: {v}')"
        ),
        make_md_cell(
            "## 2. Train Agent (Seed 0)\n\n"
            "You can reduce `total_steps` to 10000 for a quick smoke test."
        ),
        make_code_cell(
            "args = argparse.Namespace(**cfg)\n"
            "args.seed = 0\n"
            "args.tag = 'demo_nb'\n"
            "args.total_steps = 40000\n"
            "args.eval_every = 2500\n"
            "args.eval_episodes = 10\n"
            "train(args)"
        ),
        make_md_cell(
            "## 3. Visualize Training Progression\n\n"
            "Read CSV training logs from `private/logs/dqn/` and plot moving average returns."
        ),
        make_code_cell(
            "df = pd.read_csv('private/logs/dqn/demo_nb_seed0.csv')\n"
            "df['ma_return'] = df['return'].rolling(window=20, min_periods=1).mean()\n\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))\n"
            "ax1.plot(df['env_step'], df['return'], alpha=0.3, color='steelblue', label='Episode return')\n"
            "ax1.plot(df['env_step'], df['ma_return'], color='darkblue', linewidth=2, label='Moving avg (20)')\n"
            "ax1.axhline(475, color='green', linestyle='--', label='Solved threshold (475)')\n"
            "ax1.set_title('Learning Curve - Double DQN')\n"
            "ax1.set_xlabel('Environment Steps')\n"
            "ax1.set_ylabel('Return')\n"
            "ax1.grid(True, alpha=0.3)\n"
            "ax1.legend()\n\n"
            "ax2.plot(df['env_step'], df['epsilon'], color='orange', label='Epsilon decay')\n"
            "ax2.set_title('Epsilon Exploration Schedule')\n"
            "ax2.set_xlabel('Environment Steps')\n"
            "ax2.set_ylabel('Epsilon')\n"
            "ax2.grid(True, alpha=0.3)\n"
            "ax2.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        make_md_cell(
            "## 4. Evaluate Best Checkpoint on Test Seeds (100-199)"
        ),
        make_code_cell(
            "model_path = 'models/dqn_demo_nb_seed0_best.pt'\n"
            "policy = get_policy('dqn', model_path)\n"
            "eval_res = evaluate_policy(policy, n_episodes=100, seed_start=100)\n"
            "print(f'Post-training mean return: {eval_res[\"mean\"]:.2f} +- {eval_res[\"std\"]:.2f}')\n"
            "print(f'Fraction reaching max 500 score: {eval_res[\"fraction_500\"] * 100:.1f}%')"
        )
    ]
    save_notebook(cells, "notebooks/02_train_dqn.ipynb")


def generate_03_train_reinforce() -> None:
    """Generate notebooks/03_train_reinforce.ipynb to demonstrate REINFORCE training.

    Input:  None.
    Output: notebooks/03_train_reinforce.ipynb file.
    Flow:   construct markdown and code cells, invoke save_notebook.
    """
    cells = [
        make_md_cell(
            "# 03. REINFORCE (Policy Gradient) Training on CartPole-v1\n\n"
            "This notebook guides you through:\n"
            "1. Loading standard hyperparameters from `config/reinforce.yaml`.\n"
            "2. Training Policy Gradient with Return Normalization and Entropy Regularization.\n"
            "3. Visualizing training learning curves across episodes.\n"
            "4. Evaluating the best checkpoint on independent test seeds (100-199)."
        ),
        make_code_cell(
            "import os\n"
            "import sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "os.chdir('..')\n\n"
            "import argparse\n"
            "import yaml\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from libraries.reinforce import train, PolicyNetwork\n"
            "from libraries.policies import get_policy\n"
            "from libraries.evaluate import evaluate_policy"
        ),
        make_md_cell(
            "## 1. Load REINFORCE Hyperparameters"
        ),
        make_code_cell(
            "with open('config/reinforce.yaml', 'r') as f:\n"
            "    cfg = yaml.safe_load(f)\n"
            "print('Default configuration:')\n"
            "for k, v in cfg.items():\n"
            "    print(f'  {k:20s}: {v}')"
        ),
        make_md_cell(
            "## 2. Train Agent (Seed 0)"
        ),
        make_code_cell(
            "args = argparse.Namespace(**cfg)\n"
            "args.seed = 0\n"
            "args.tag = 'demo_nb'\n"
            "args.total_steps = 40000\n"
            "args.eval_every = 2500\n"
            "args.eval_episodes = 10\n"
            "train(args)"
        ),
        make_md_cell(
            "## 3. Visualize Training Progression"
        ),
        make_code_cell(
            "df = pd.read_csv('private/logs/reinforce/demo_nb_seed0.csv')\n"
            "df['ma_return'] = df['return'].rolling(window=20, min_periods=1).mean()\n\n"
            "plt.figure(figsize=(9, 4.5))\n"
            "plt.plot(df['env_step'], df['return'], alpha=0.3, color='forestgreen', label='Episode return')\n"
            "plt.plot(df['env_step'], df['ma_return'], color='darkgreen', linewidth=2, label='Moving avg (20)')\n"
            "plt.axhline(475, color='red', linestyle='--', label='Solved threshold (475)')\n"
            "plt.title('Learning Curve - REINFORCE (Policy Gradient)')\n"
            "plt.xlabel('Environment Steps')\n"
            "plt.ylabel('Return')\n"
            "plt.grid(True, alpha=0.3)\n"
            "plt.legend()\n"
            "plt.show()"
        ),
        make_md_cell(
            "## 4. Evaluate Best Checkpoint on Test Seeds (100-199)"
        ),
        make_code_cell(
            "model_path = 'models/reinforce_demo_nb_seed0_best.pt'\n"
            "policy = get_policy('reinforce', model_path)\n"
            "eval_res = evaluate_policy(policy, n_episodes=100, seed_start=100)\n"
            "print(f'Post-training mean return: {eval_res[\"mean\"]:.2f} +- {eval_res[\"std\"]:.2f}')\n"
            "print(f'Fraction reaching max 500 score: {eval_res[\"fraction_500\"] * 100:.1f}%')"
        )
    ]
    save_notebook(cells, "notebooks/03_train_reinforce.ipynb")


def generate_05_analysis() -> None:
    """Generate notebooks/05_analysis.ipynb for comparative analysis and error diagnosis.

    Input:  None.
    Output: notebooks/05_analysis.ipynb file.
    Flow:   construct markdown and code cells, invoke save_notebook.
    """
    cells = [
        make_md_cell(
            "# 05. Comparative Analysis, Robustness, and Failure Diagnosis\n\n"
            "This notebook guides you through:\n"
            "1. Inspecting the aggregated summary table comparing Random, Double DQN, and REINFORCE.\n"
            "2. Analyzing robustness under observation noise.\n"
            "3. In-depth failure analysis: state trajectories and policy boundaries."
        ),
        make_code_cell(
            "import os\n"
            "import sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "os.chdir('..')\n\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from IPython.display import Image, display"
        ),
        make_md_cell(
            "## 1. Experimental Summary Table (summary.csv)\n\n"
            "Metrics aggregated across 5 independent seeds evaluated on test seeds 100-199."
        ),
        make_code_cell(
            "df_summary = pd.read_csv('outputs/results/summary.csv')\n"
            "cols = ['algo', 'tag', 'n_seeds', 'after_best_mean', 'after_best_std_across_seeds', 'frac500', 'after_final_mean', 'n_seeds_ge475']\n"
            "display(df_summary[cols])"
        ),
        make_md_cell(
            "## 2. Compare Learning Curves"
        ),
        make_code_cell(
            "if os.path.exists('outputs/figures/learning_curves.png'):\n"
            "    display(Image(filename='outputs/figures/learning_curves.png'))\n"
            "else:\n"
            "    print('outputs/figures/learning_curves.png not found, run python -m libraries.plot_curves')"
        ),
        make_md_cell(
            "## 3. Observation Noise Robustness\n\n"
            "Agents trained on standard CartPole (V0) and tested under Gaussian noise (V2) with $\\sigma \\in \\{0.0, 0.01, 0.05, 0.1\\}$."
        ),
        make_code_cell(
            "df_noise = pd.read_csv('outputs/results/noise_robustness.csv')\n"
            "print('Noise robustness measurements:')\n"
            "display(df_noise.pivot_table(index='sigma', columns='algo', values=['mean', 'std']))\n\n"
            "if os.path.exists('outputs/figures/noise_robustness.png'):\n"
            "    display(Image(filename='outputs/figures/noise_robustness.png'))"
        ),
        make_md_cell(
            "## 4. Error and Failure Mode Analysis\n\n"
            "Inspect state trajectories during early failures and policy decision boundaries."
        ),
        make_code_cell(
            "if os.path.exists('outputs/figures/trajectories.png'):\n"
            "    print('State trajectories (x, theta) over time:')\n"
            "    display(Image(filename='outputs/figures/trajectories.png'))\n\n"
            "if os.path.exists('outputs/figures/policy_heatmap.png'):\n"
            "    print('Action decision boundary across angle and angular velocity:')\n"
            "    display(Image(filename='outputs/figures/policy_heatmap.png'))"
        )
    ]
    save_notebook(cells, "notebooks/05_analysis.ipynb")


def main() -> None:
    """CLI entrypoint: generate all project Jupyter notebooks.

    Input:  None.
    Output: generated ipynb files in notebooks/ directory.
    Flow:   invoke notebook generation functions in sequence, print completion messages.
    """
    generate_01_explore()
    generate_02_train_dqn()
    generate_03_train_reinforce()
    generate_05_analysis()
    print("Successfully generated all notebooks:")
    print("  - notebooks/01_explore_env.ipynb")
    print("  - notebooks/02_train_dqn.ipynb")
    print("  - notebooks/03_train_reinforce.ipynb")
    print("  - notebooks/05_analysis.ipynb")


if __name__ == "__main__":
    main()

