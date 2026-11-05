"""REINFORCE (Policy Gradient) for CartPole-v1 with discounted return normalization and entropy bonus.

Single seed:  python -m libraries.reinforce --seed 0 --tag base
Smoke test:   python -m libraries.reinforce --seed 0 --tag smoke --total_steps 5000 --eval_every 2500 --eval_episodes 3

Shares identical output formatting and log columns with dqn.py (algo="reinforce"), with epsilon blank.
Default budget is 150_000 environment steps, matching DQN for fair benchmark comparison.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .envs import make_env
from .evaluate import evaluate_policy
from .utils import (DEFAULT_LOG_DIR, SELECT_SEED_START, TEST_SEED_START, CsvLogger,
                    ensure_dir, save_json, set_seed)

torch.set_num_threads(1)

DEFAULT_RESULTS_DIR = "outputs/results"
DEFAULT_SEED = 0
DEFAULT_TAG = "base"
DEFAULT_TOTAL_STEPS = 150_000
DEFAULT_LR = 3e-3
DEFAULT_GAMMA = 0.99
DEFAULT_HIDDEN = 128
DEFAULT_ENT_COEF = 0.005
DEFAULT_EPISODES_PER_UPDATE = 1
DEFAULT_NO_NORMALIZE = False
DEFAULT_EVAL_EVERY = 5_000
DEFAULT_EVAL_EPISODES = 20
DEFAULT_TRAIN_VARIANT = "v0"
DEFAULT_SIGMA = 0.05


class PolicyNetwork(nn.Module):
    """MLP parameterizing discrete action distribution with raw unnormalized logits.

    Input (init): obs_dim   -- observation dimension (default 4);
                  n_actions -- discrete action count (default 2);
                  hidden    -- hidden layer neuron count (default 128).
    Output forward: float32 tensor of shape (batch, n_actions) containing action logits.
    Flow:           linear(4, hidden) -> ReLU -> linear(hidden, n_actions).
    """

    def __init__(self, obs_dim: int = 4, n_actions: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        """Compute action logits for an observation batch.

        Input:  x -- float32 tensor with shape (batch, obs_dim).
        Output: float32 tensor with shape (batch, n_actions) containing logits.
        Flow:   forward pass through sequential MLP layers.
        """
        return self.net(x)


def rollout(env, policy, first_seed=None):
    """Execute a complete on-policy rollout episode and record action log-probabilities, entropies, and rewards.

    Input:  env        -- Gymnasium environment;
            policy     -- PolicyNetwork instance;
            first_seed -- optional initial reset seed.
    Output: tuple (logps, ents, rews, raw) where:
            logps -- list of scalar log-probability tensors;
            ents  -- list of scalar entropy tensors;
            rews  -- list of step reward floats;
            raw   -- accumulated unshaped raw return float.
    Flow:   reset env, iterate steps: sample action from Categorical distribution, step env,
            collect log_prob, entropy, step reward, accumulate raw return until episode end.
    """
    obs, _ = env.reset(seed=first_seed)
    logps, ents, rews = [], [], []
    raw, done = 0.0, False
    while not done:
        dist = Categorical(logits=policy(torch.as_tensor(obs, dtype=torch.float32)))
        a = dist.sample()
        obs, r, terminated, truncated, info = env.step(int(a.item()))
        logps.append(dist.log_prob(a))
        ents.append(dist.entropy())
        rews.append(r)
        raw += info.get("raw_reward", r)
        done = terminated or truncated
    return logps, ents, rews, raw


def discounted_returns(rews, gamma):
    """Compute discounted returns G_t = sum_{k=0} gamma^k * r_{t+k} via backward dynamic programming.

    Input:  rews  -- list of reward floats in chronological order;
            gamma -- discount factor float.
    Output: list of discounted return floats matching input length.
    Flow:   traverse rewards in reverse, accumulate discounted return, reverse back to chronological order.
    """
    G, out = 0.0, []
    for r in reversed(rews):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return out


def save_policy(policy, hidden: int, path: str) -> None:
    """Save PolicyNetwork weights and metadata to a .pt file.

    Input:  policy -- PolicyNetwork instance;
            hidden -- hidden dimension integer;
            path   -- target checkpoint path.
    Output: None (writes to disk).
    Flow:   ensure directory exists, call torch.save with algo, hidden, and state_dict.
    """
    ensure_dir(os.path.dirname(path))
    torch.save({"algo": "reinforce", "hidden": hidden,
                "state_dict": policy.state_dict()}, path)


def load_into(policy, path: str) -> None:
    """Load weights from a saved checkpoint .pt file into an existing PolicyNetwork.

    Input:  policy -- destination PolicyNetwork instance;
            path   -- saved checkpoint path.
    Output: None (updates model state in place).
    Flow:   call torch.load, invoke load_state_dict.
    """
    policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])


def train(args=None, **kwargs) -> None:
    """Execute main REINFORCE training loop over total_steps environment steps.

    Input:  args   -- Namespace of hyperparameters or None;
            kwargs -- optional hyperparameter overrides.
    Output: None (saves CSV logs, .pt checkpoints, and JSON test results).
    Flow:   instantiate policy network, roll out full episodes, compute discounted returns,
            normalize advantages, compute policy gradient loss with entropy bonus,
            update weights via Adam, periodically evaluate and save checkpoints.
    """
    if args is None:
        cfg = {
            "seed": DEFAULT_SEED, "tag": DEFAULT_TAG, "total_steps": DEFAULT_TOTAL_STEPS,
            "lr": DEFAULT_LR, "gamma": DEFAULT_GAMMA, "hidden": DEFAULT_HIDDEN,
            "ent_coef": DEFAULT_ENT_COEF, "episodes_per_update": DEFAULT_EPISODES_PER_UPDATE,
            "no_normalize": DEFAULT_NO_NORMALIZE, "eval_every": DEFAULT_EVAL_EVERY,
            "eval_episodes": DEFAULT_EVAL_EPISODES, "train_variant": DEFAULT_TRAIN_VARIANT,
            "sigma": DEFAULT_SIGMA,
        }
        cfg.update(kwargs)
        args = argparse.Namespace(**cfg)
    elif kwargs:
        for k, v in kwargs.items():
            setattr(args, k, v)

    set_seed(args.seed)
    run = f"{args.tag}_seed{args.seed}"
    best_path = f"models/reinforce_{run}_best.pt"
    final_path = f"models/reinforce_{run}_final.pt"

    env = make_env(args.train_variant, seed=args.seed, sigma=args.sigma)
    policy = PolicyNetwork(hidden=args.hidden)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    def greedy(obs):
        """Greedy deterministic action selector from current policy network.

        Input:  obs -- float32 observation array of shape (obs_dim,).
        Output: action integer corresponding to argmax logit.
        Flow:   forward pass through policy network, take argmax of logits.
        """
        with torch.no_grad():
            return int(policy(torch.as_tensor(obs, dtype=torch.float32)).argmax().item())

    train_log = CsvLogger(f"{DEFAULT_LOG_DIR}/reinforce/{run}.csv",
                          ["algo", "seed", "episode", "env_step", "return", "length", "epsilon", "loss"])
    eval_log = CsvLogger(f"{DEFAULT_LOG_DIR}/reinforce/{run}_eval.csv",
                         ["algo", "seed", "env_step", "eval_mean", "eval_std", "eval_min", "eval_max"])
    save_json(vars(args), f"{DEFAULT_LOG_DIR}/reinforce/{run}_config.json")

    before = evaluate_policy(greedy, n_episodes=100, seed_start=TEST_SEED_START)
    print(f"[reinforce seed={args.seed}] BEFORE training: mean={before['mean']:.1f}")

    t0 = time.time()
    t, ep, best, next_eval = 0, 0, -1.0, args.eval_every
    while t < args.total_steps:
        all_logps, all_ents, all_G, batch_stats = [], [], [], []
        for _ in range(args.episodes_per_update):
            logps, ents, rews, raw = rollout(env, policy, first_seed=args.seed if ep == 0 else None)
            ep += 1
            t += len(rews)
            all_logps += logps
            all_ents += ents
            all_G += discounted_returns(rews, args.gamma)
            batch_stats.append((ep, t, raw, len(rews)))

        G = torch.tensor(all_G, dtype=torch.float32)
        if not args.no_normalize:
            G = (G - G.mean()) / (G.std() + 1e-8)
        loss = -(torch.stack(all_logps) * G).sum() - args.ent_coef * torch.stack(all_ents).sum()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        opt.step()

        for (e, step, raw, ln) in batch_stats:
            train_log.log({"algo": "reinforce", "seed": args.seed, "episode": e, "env_step": step,
                           "return": raw, "length": ln, "epsilon": "", "loss": float(loss.item())})

        if t >= next_eval:
            res = evaluate_policy(greedy, n_episodes=args.eval_episodes, seed_start=SELECT_SEED_START)
            eval_log.log({"algo": "reinforce", "seed": args.seed, "env_step": t,
                          "eval_mean": res["mean"], "eval_std": res["std"],
                          "eval_min": res["min"], "eval_max": res["max"]})
            if res["mean"] >= best:
                best = res["mean"]
                save_policy(policy, args.hidden, best_path)
            print(f"[reinforce seed={args.seed}] step={t} ep={ep} eval_mean={res['mean']:.1f} best={best:.1f}")
            next_eval = (t // args.eval_every + 1) * args.eval_every

    save_policy(policy, args.hidden, final_path)
    if not os.path.exists(best_path):
        save_policy(policy, args.hidden, best_path)
    train_log.close()
    eval_log.close()

    load_into(policy, best_path)
    after_best = evaluate_policy(greedy, n_episodes=100, seed_start=TEST_SEED_START)
    load_into(policy, final_path)
    after_final = evaluate_policy(greedy, n_episodes=100, seed_start=TEST_SEED_START)
    print(f"[reinforce seed={args.seed}] AFTER training: best={after_best['mean']:.1f} final={after_final['mean']:.1f} "
          f"({time.time() - t0:.0f}s)")

    save_json({"algo": "reinforce", "tag": args.tag, "seed": args.seed, "config": vars(args),
               "before": before, "after_best": after_best, "after_final": after_final,
               "wall_time_s": time.time() - t0},
              f"{DEFAULT_RESULTS_DIR}/reinforce_{run}.json")


def build_parser():
    """Build command line argument parser for reinforce.py.

    Input:  None.
    Output: configured argparse.ArgumentParser instance.
    Flow:   register arguments with defaults drawn from module-level constants.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--total_steps", type=int, default=DEFAULT_TOTAL_STEPS)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    ap.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    ap.add_argument("--ent_coef", type=float, default=DEFAULT_ENT_COEF)
    ap.add_argument("--episodes_per_update", type=int, default=DEFAULT_EPISODES_PER_UPDATE)
    ap.add_argument("--no_normalize", action="store_true", default=DEFAULT_NO_NORMALIZE)
    ap.add_argument("--eval_every", type=int, default=DEFAULT_EVAL_EVERY)
    ap.add_argument("--eval_episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    ap.add_argument("--train_variant", default=DEFAULT_TRAIN_VARIANT, choices=["v0", "v1", "v2", "v3"])
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    return ap


if __name__ == "__main__":
    train(build_parser().parse_args())
