"""Double DQN for CartPole-v1 with experience replay buffer and soft target network updates.

Single seed:   python -m libraries.dqn --seed 0 --tag base
Smoke test:    python -m libraries.dqn --seed 0 --tag smoke --total_steps 5000 --eval_every 2500 --eval_episodes 3

Outputs (tag=<tag>, seed=<k>):
  private/logs/dqn/<tag>_seed<k>.csv          per-episode training logs
  private/logs/dqn/<tag>_seed<k>_eval.csv     periodic evaluation logs (seeds 1000..1019, epsilon=0)
  private/logs/dqn/<tag>_seed<k>_config.json
  models/dqn_<tag>_seed<k>_best.pt / _final.pt
  outputs/results/dqn_<tag>_seed<k>.json      pre- and post-training test returns on seeds 100..199
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .envs import make_env
from .evaluate import evaluate_policy
from .utils import (DEFAULT_LOG_DIR, SELECT_SEED_START, TEST_SEED_START, CsvLogger,
                    ensure_dir, save_json, set_seed)

torch.set_num_threads(1)

DEFAULT_RESULTS_DIR = "outputs/results"
DEFAULT_SEED = 0
DEFAULT_TAG = "base"
DEFAULT_TOTAL_STEPS = 150_000
DEFAULT_LR = 1e-3
DEFAULT_GAMMA = 0.99
DEFAULT_BUFFER = 50_000
DEFAULT_BATCH = 64
DEFAULT_HIDDEN = 128
DEFAULT_LEARNING_STARTS = 1_000
DEFAULT_EPS_END = 0.05
DEFAULT_EPS_DECAY = 10_000
DEFAULT_UPDATE = "soft"
DEFAULT_TAU = 0.005
DEFAULT_HARD_EVERY = 500
DEFAULT_NO_DOUBLE = False
DEFAULT_EVAL_EVERY = 5_000
DEFAULT_EVAL_EPISODES = 20
DEFAULT_TRAIN_VARIANT = "v0"
DEFAULT_SIGMA = 0.05


class QNetwork(nn.Module):
    """MLP architecture estimating state-action values Q(s, a).

    Input (init): obs_dim   -- observation dimension (default 4);
                  n_actions -- discrete action count (default 2);
                  hidden    -- hidden layer neuron count (default 128).
    Output forward: raw unnormalized Q-value tensor of shape (batch, n_actions).
    Flow:           linear(4, hidden) -> ReLU -> linear(hidden, hidden) -> ReLU -> linear(hidden, n_actions).
    """

    def __init__(self, obs_dim: int = 4, n_actions: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        """Compute Q-values for an observation batch.

        Input:  x -- float32 tensor with shape (batch, obs_dim).
        Output: float32 tensor with shape (batch, n_actions).
        Flow:   forward pass through sequential MLP layers.
        """
        return self.net(x)


class ReplayBuffer:
    """Ring buffer storing transition tuples (s, a, r, s2, terminated).

    Input (init): capacity -- maximum transitions stored;
                  obs_dim  -- observation dimension (default 4).
    Flow:         pre-allocate numpy buffers, advance circular write pointer.
    """

    def __init__(self, capacity: int, obs_dim: int = 4):
        self.cap, self.n, self.i = capacity, 0, 0
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros((capacity, 1), np.int64)
        self.r = np.zeros((capacity, 1), np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.d = np.zeros((capacity, 1), np.float32)

    def push(self, s, a, r, s2, d) -> None:
        """Append a new transition to the buffer.

        Input:  s  -- current observation (ndarray float32 of shape (obs_dim,));
                a  -- action integer;
                r  -- immediate reward float;
                s2 -- next observation (ndarray float32 of shape (obs_dim,));
                d  -- float(terminated), never merged with truncated.
        Output: None (mutates internal arrays).
        Flow:   write transition into slot self.i, advance circular pointer, increment count.
        """
        i = self.i
        self.s[i], self.a[i, 0], self.r[i, 0], self.s2[i], self.d[i, 0] = s, a, r, s2, d
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch: int):
        """Uniformly sample a batch of transitions.

        Input:  batch -- number of transitions to sample.
        Output: 5-tuple of PyTorch tensors (s, a, r, s2, d).
        Flow:   draw random indices, slice numpy arrays, convert to tensors.
        """
        idx = np.random.randint(0, self.n, size=batch)
        return tuple(torch.from_numpy(x[idx]) for x in (self.s, self.a, self.r, self.s2, self.d))

    def __len__(self):
        """Return current transition count in the buffer.

        Input:  None.
        Output: integer current buffer size self.n.
        Flow:   return self.n.
        """
        return self.n


class DQNAgent:
    """Double DQN agent managing online and target networks, optimizer, and Bellman updates.

    Input (init): args -- Namespace containing configuration hyperparameters.
    Flow:         instantiate online and target QNetworks, synchronize weights, initialize Adam.
    """

    def __init__(self, args):
        self.hidden = args.hidden
        self.gamma = args.gamma
        self.double = not args.no_double
        self.update_mode = args.update
        self.tau = args.tau
        self.hard_every = args.hard_every
        self.learn_steps = 0
        self.online = QNetwork(hidden=args.hidden)
        self.target = QNetwork(hidden=args.hidden)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=args.lr)

    def greedy(self, obs) -> int:
        """Select greedy action with argmax Q under no_grad context.

        Input:  obs -- float32 observation array of shape (obs_dim,).
        Output: action integer with highest estimated Q-value.
        Flow:   cast obs to tensor, forward through online network, return argmax index.
        """
        with torch.no_grad():
            return int(self.online(torch.as_tensor(obs, dtype=torch.float32)).argmax().item())

    def learn(self, buf: ReplayBuffer, batch: int) -> float:
        """Perform one Double DQN gradient update and update the target network.

        Input:  buf   -- ReplayBuffer with at least batch transitions;
                batch -- minibatch sample size.
        Output: float Huber loss value for this step.
        Flow:   sample batch, compute Q_online(s, a), compute Bellman target y with Double Q logic,
                calculate Huber loss, clip gradient norm to 10.0, step optimizer, update target network.
        """
        s, a, r, s2, d = buf.sample(batch)
        q = self.online(s).gather(1, a)
        with torch.no_grad():
            if self.double:
                a2 = self.online(s2).argmax(1, keepdim=True)
                q2 = self.target(s2).gather(1, a2)
            else:
                q2 = self.target(s2).max(1, keepdim=True)[0]
            y = r + self.gamma * (1.0 - d) * q2
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        self.learn_steps += 1
        if self.update_mode == "soft":
            with torch.no_grad():
                for p, tp in zip(self.online.parameters(), self.target.parameters()):
                    tp.mul_(1.0 - self.tau).add_(self.tau * p)
        elif self.learn_steps % self.hard_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path: str) -> None:
        """Save online network weights and metadata to a .pt file.

        Input:  path -- target checkpoint file path.
        Output: None (writes to disk).
        Flow:   ensure directory exists, call torch.save with algo, hidden, and state_dict.
        """
        ensure_dir(os.path.dirname(path))
        torch.save({"algo": "dqn", "hidden": self.hidden,
                    "state_dict": self.online.state_dict()}, path)

    def load(self, path: str) -> None:
        """Load network weights from a .pt file into both online and target networks.

        Input:  path -- saved checkpoint file path.
        Output: None (updates model state in place).
        Flow:   call torch.load, load state_dict into online and target networks.
        """
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.online.load_state_dict(ck["state_dict"])
        self.target.load_state_dict(ck["state_dict"])


def train(args=None, **kwargs) -> None:
    """Execute main Double DQN training loop over total_steps environment steps.

    Input:  args   -- Namespace of hyperparameters or None;
            kwargs -- optional hyperparameter overrides.
    Output: None (saves CSV logs, .pt checkpoints, and JSON test results).
    Flow:   initialize agent and replay buffer, iterate through environment steps,
            decay epsilon, step environment, sample and train on minibatches,
            periodically evaluate and save checkpoints, evaluate on test seeds post-training.
    """
    if args is None:
        cfg = {
            "seed": DEFAULT_SEED, "tag": DEFAULT_TAG, "total_steps": DEFAULT_TOTAL_STEPS,
            "lr": DEFAULT_LR, "gamma": DEFAULT_GAMMA, "buffer": DEFAULT_BUFFER,
            "batch": DEFAULT_BATCH, "hidden": DEFAULT_HIDDEN,
            "learning_starts": DEFAULT_LEARNING_STARTS, "eps_end": DEFAULT_EPS_END,
            "eps_decay": DEFAULT_EPS_DECAY, "update": DEFAULT_UPDATE, "tau": DEFAULT_TAU,
            "hard_every": DEFAULT_HARD_EVERY, "no_double": DEFAULT_NO_DOUBLE,
            "eval_every": DEFAULT_EVAL_EVERY, "eval_episodes": DEFAULT_EVAL_EPISODES,
            "train_variant": DEFAULT_TRAIN_VARIANT, "sigma": DEFAULT_SIGMA,
        }
        cfg.update(kwargs)
        args = argparse.Namespace(**cfg)
    elif kwargs:
        for k, v in kwargs.items():
            setattr(args, k, v)

    set_seed(args.seed)
    run = f"{args.tag}_seed{args.seed}"
    best_path = f"models/dqn_{run}_best.pt"
    final_path = f"models/dqn_{run}_final.pt"

    env = make_env(args.train_variant, seed=args.seed, sigma=args.sigma)
    agent = DQNAgent(args)
    buf = ReplayBuffer(args.buffer)

    train_log = CsvLogger(f"{DEFAULT_LOG_DIR}/dqn/{run}.csv",
                          ["algo", "seed", "episode", "env_step", "return", "length", "epsilon", "loss"])
    eval_log = CsvLogger(f"{DEFAULT_LOG_DIR}/dqn/{run}_eval.csv",
                         ["algo", "seed", "env_step", "eval_mean", "eval_std", "eval_min", "eval_max"])
    save_json(vars(args), f"{DEFAULT_LOG_DIR}/dqn/{run}_config.json")

    before = evaluate_policy(agent.greedy, n_episodes=100, seed_start=TEST_SEED_START)
    print(f"[dqn seed={args.seed}] BEFORE training: mean={before['mean']:.1f}")

    t0 = time.time()
    obs, _ = env.reset(seed=args.seed)
    ep, ep_ret, ep_len, losses = 0, 0.0, 0, []
    best, next_eval = -1.0, args.eval_every

    for t in range(1, args.total_steps + 1):
        eps = max(args.eps_end, 1.0 - (1.0 - args.eps_end) * t / args.eps_decay)
        a = random.randrange(2) if random.random() < eps else agent.greedy(obs)
        obs2, r, terminated, truncated, info = env.step(a)
        buf.push(obs, a, r, obs2, float(terminated))
        ep_ret += info.get("raw_reward", r)
        ep_len += 1
        obs = obs2

        if len(buf) >= args.learning_starts:
            losses.append(agent.learn(buf, args.batch))

        if terminated or truncated:
            ep += 1
            train_log.log({"algo": "dqn", "seed": args.seed, "episode": ep, "env_step": t,
                           "return": ep_ret, "length": ep_len, "epsilon": round(eps, 4),
                           "loss": float(np.mean(losses)) if losses else ""})
            obs, _ = env.reset()
            ep_ret, ep_len, losses = 0.0, 0, []

        if t >= next_eval:
            res = evaluate_policy(agent.greedy, n_episodes=args.eval_episodes,
                                  seed_start=SELECT_SEED_START)
            eval_log.log({"algo": "dqn", "seed": args.seed, "env_step": t,
                          "eval_mean": res["mean"], "eval_std": res["std"],
                          "eval_min": res["min"], "eval_max": res["max"]})
            if res["mean"] >= best:
                best = res["mean"]
                agent.save(best_path)
            print(f"[dqn seed={args.seed}] step={t} eps={eps:.2f} eval_mean={res['mean']:.1f} best={best:.1f}")
            next_eval += args.eval_every

    agent.save(final_path)
    if not os.path.exists(best_path):
        agent.save(best_path)
    train_log.close()
    eval_log.close()

    agent.load(best_path)
    after_best = evaluate_policy(agent.greedy, n_episodes=100, seed_start=TEST_SEED_START)
    agent.load(final_path)
    after_final = evaluate_policy(agent.greedy, n_episodes=100, seed_start=TEST_SEED_START)
    print(f"[dqn seed={args.seed}] AFTER training: best={after_best['mean']:.1f} final={after_final['mean']:.1f} "
          f"({time.time() - t0:.0f}s)")

    save_json({"algo": "dqn", "tag": args.tag, "seed": args.seed, "config": vars(args),
               "before": before, "after_best": after_best, "after_final": after_final,
               "wall_time_s": time.time() - t0},
              f"{DEFAULT_RESULTS_DIR}/dqn_{run}.json")


def build_parser():
    """Build command line argument parser for dqn.py.

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
    ap.add_argument("--buffer", type=int, default=DEFAULT_BUFFER)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    ap.add_argument("--learning_starts", type=int, default=DEFAULT_LEARNING_STARTS)
    ap.add_argument("--eps_end", type=float, default=DEFAULT_EPS_END)
    ap.add_argument("--eps_decay", type=int, default=DEFAULT_EPS_DECAY)
    ap.add_argument("--update", choices=["soft", "hard"], default=DEFAULT_UPDATE)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--hard_every", type=int, default=DEFAULT_HARD_EVERY)
    ap.add_argument("--no_double", action="store_true", default=DEFAULT_NO_DOUBLE)
    ap.add_argument("--eval_every", type=int, default=DEFAULT_EVAL_EVERY)
    ap.add_argument("--eval_episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    ap.add_argument("--train_variant", default=DEFAULT_TRAIN_VARIANT, choices=["v0", "v1", "v2", "v3"])
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    return ap


if __name__ == "__main__":
    train(build_parser().parse_args())
