# CartPole Reinforcement Learning

> Training and evaluating reinforcement learning agents on CartPole-v1 using Double DQN (primary) and REINFORCE (baseline control).

## Introduction

This project trains, benchmarks, and analyzes three policies on the Gymnasium CartPole-v1 control task:
- **Random Policy**: Random baseline acting uniformly at random (mean return ~20-25 steps).
- **Double DQN**: Primary value-based method with replay buffer and target network (target score >= 475/500 steps).
- **REINFORCE**: Policy gradient control method with discounted return normalization and entropy regularization.

## Directory Structure

```
libraries/    : Core Python package (envs, evaluate, policies, dqn, reinforce, utils, summarize)
scripts/      : Automation scripts (check_gate.py, run_seeds.py, generate_notebooks.py, make_demo_notebook.py)
tests/        : Unit tests (test_dqn_units.py)
models/       : Saved neural network checkpoints (.pt)
outputs/      : Generated figures (outputs/figures/), demo videos (outputs/videos/), results (outputs/results/)
public/       : Public English docs (public/docs/), modular live web UI (public/web/{index.html,css,js})
private/      : Internal Vietnamese docs (private/docs/) and training logs (private/logs/); gitignored
notebooks/    : Interactive Jupyter notebooks (01_explore through 05_analysis)
config/       : Algorithm hyperparameters (dqn.yaml, reinforce.yaml)
```

## Installation

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux or macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## Environment Verification

Verify that Gymnasium, PyTorch, and image rendering work properly:

```bash
python -c "import gymnasium as gym; e=gym.make('CartPole-v1', render_mode='rgb_array'); o,_=e.reset(seed=0); print(o); print(e.render().shape)"
```

## Project Workflow (Phased Execution)

### Phase 0-1: Setup and Random Baseline

```bash
python -m libraries.explore_env
python -m libraries.baseline --policy random
python -m libraries.baseline --policy heuristic
python -m libraries.record_gif --algo random --out outputs/videos/random.gif --seed 3000
python scripts/check_gate.py 1
```

### Phase 2: Double DQN

```bash
# Smoke test
python -m libraries.dqn --seed 0 --tag smoke --total_steps 5000 --eval_every 2500 --eval_episodes 3

# Single seed run
python -m libraries.dqn --seed 0 --tag base

# Multi-seed run (seeds 0 through 4)
python scripts/run_seeds.py dqn base

python scripts/check_gate.py 2
```

### Phase 3: REINFORCE

```bash
python -m libraries.reinforce --seed 0 --tag smoke --total_steps 5000 --eval_every 2500 --eval_episodes 3
python scripts/run_seeds.py reinforce base
python scripts/check_gate.py 3
```

### Phase 4: Comparison and Ablation

```bash
python -m libraries.summarize
python -m libraries.plot_curves --runs dqn:base reinforce:base --out outputs/figures/learning_curves.png
python -m libraries.robustness --runs dqn:base reinforce:base --sigmas 0 0.01 0.05 0.1
python scripts/check_gate.py 4
```

### Phase 5: Demo and Error Analysis

```bash
python -m libraries.record_gif --algo dqn --model models/dqn_base_seed0_best.pt --out outputs/videos/dqn_best.gif --seed 3000
python -m libraries.record_gif --algo reinforce --model models/reinforce_base_seed0_best.pt --out outputs/videos/reinforce_best.gif --seed 3000
python -m libraries.record_gif --algo dqn --model models/dqn_base_seed0_best.pt --out outputs/videos/dqn_new_data.gif --seed 3001
python -m libraries.error_analysis --algo dqn --model models/dqn_base_seed0_best.pt
python scripts/make_demo_notebook.py --algo dqn --model models/dqn_base_seed0_best.pt
python scripts/check_gate.py 5
```

### Phase 6: Report and Artifacts

```bash
python scripts/check_gate.py 6
```

## Real-Time Web Streaming (Infinite Execution)

Run an interactive real-time web stream that executes indefinitely with live physics visualization and controls:

```bash
# Default: Double DQN in infinite horizon mode at 30 FPS
python scripts/run_stream.py

# Custom algorithm, port, and simulation speed:
python scripts/run_stream.py --port 8000 --algo dqn --mode infinite --fps 30
python scripts/run_stream.py --algo reinforce --mode standard
python -m libraries.stream --algo heuristic
```

Open `http://localhost:8000` in any web browser (UI from `public/web/`: HTML + `css/` + `js/`) to:
- Watch live MJPEG video rendering at full framerate (`/stream`).
- Run indefinitely with continuous balancing (no 500-step ceiling) or auto-resetting multi-episode benchmarking.
- Inspect real-time state telemetry: pole angle gauge, cart position rail, and Q-values / action probabilities.
- View real-time HTML5 Canvas waveform showing angle and position history.
- Inject physical disturbances: push cart left or right, tilt pole, or trigger random wind impulses to test agent recovery.
- Switch policies on the fly: Double DQN, REINFORCE, Heuristic, Random baseline, or Manual keyboard control.

## Running Unit Tests

```bash
python -m pytest tests/test_dqn_units.py -v
```

## Running All Acceptance Gates

```bash
python scripts/check_gate.py all
```

## Seed Conventions

| Purpose | Seed Range | Description |
|---|---|---|
| Training | 0, 1, 2, 3, 4 | Independent environment and initialization seeds |
| Checkpoint selection | 1000-1019 | Periodic evaluation during training to choose best model |
| Final test evaluation | 100-199 | 100 independent evaluation episodes (never seen during training) |
| Error analysis | 2000-2499 | 500 evaluation episodes for trajectory and failure analysis |
| Novel data demo | 3000+ | Unseen test environments for video rendering |

## Core Principles

1. Never report unverified or fake results.
2. Never use test seeds (100-199) for hyperparameter tuning or checkpoint selection.
3. Report mean +/- standard deviation across 5 seeds.
4. Evaluation returns are always raw environment steps, even when training on reward-shaped variants.
5. Only `terminated` cuts bootstrapping in DQN, never `truncated`.

## Code Standards and Contributing

- Code comment and docstring standards: [COMMENT.md](COMMENT.md)
- Contribution guidelines: [CONTRIBUTING.md](CONTRIBUTING.md)
- Security policy: [SECURITY.md](SECURITY.md)
- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## License

This project is licensed under the [MIT License](LICENSE).
Copyright (c) 2026 orias <orias@orias.app>.
