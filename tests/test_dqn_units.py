"""Unit tests for critical Double DQN components.

Execution:
    pytest tests/test_dqn_units.py -v
"""
import argparse

import numpy as np
import torch

from libraries.dqn import DQNAgent, QNetwork, ReplayBuffer, build_parser


def make_args(**kw):
    """Create hyperparameter Namespace with default values and custom overrides.

    Input:  **kw -- mapping of hyperparameter name to override value.
    Output: complete argparse.Namespace of hyperparameters.
    Flow:   invoke build_parser().parse_args([]), setattr for each override keyword argument.
    """
    args = build_parser().parse_args([])
    for k, v in kw.items():
        setattr(args, k, v)
    return args


def test_qnetwork_shape():
    """Verify QNetwork output tensor shape corresponds to (batch, n_actions).

    Input:  None.
    Output: assertion passes without exception.
    Flow:   pass zero batch of 7 observation vectors through network, assert shape (7, 2).
    """
    out = QNetwork()(torch.zeros(7, 4))
    assert out.shape == (7, 2)


def test_buffer_shapes_and_terminated_flag():
    """Verify circular replay buffer capacity, sample shapes, and binary terminated flag.

    Input:  None.
    Output: assertions pass without exception.
    Flow:   push 150 transitions into a buffer of capacity 100, draw batch of 32,
            verify tensor shapes and dtypes, assert termination flag belongs to {0.0, 1.0}.
    """
    buf = ReplayBuffer(100)
    for i in range(150):
        buf.push(np.ones(4) * i, i % 2, 1.0, np.ones(4) * (i + 1), float(i % 10 == 0))
    assert len(buf) == 100
    s, a, r, s2, d = buf.sample(32)
    assert s.shape == (32, 4) and a.shape == (32, 1) and r.shape == (32, 1)
    assert s2.shape == (32, 4) and d.shape == (32, 1)
    assert set(d.unique().tolist()) <= {0.0, 1.0}
    assert a.dtype == torch.int64


def test_learning_reduces_loss_on_fixed_data():
    """Overfitting unit test: verify loss decreases significantly on fixed synthetic data.

    Input:  None.
    Output: assertion passes verifying final loss < 0.2 * initial loss.
    Flow:   seed RNGs, construct synthetic dataset with terminal rewards, train agent for 400 steps,
            compare initial average loss against final average loss.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    agent = DQNAgent(make_args(lr=1e-2, update="hard", hard_every=10**9))
    buf = ReplayBuffer(256)
    rng = np.random.default_rng(0)
    for _ in range(256):
        s = rng.normal(size=4).astype(np.float32)
        buf.push(s, int(rng.integers(0, 2)), 1.0, s, 1.0)
    first = np.mean([agent.learn(buf, 64) for _ in range(5)])
    for _ in range(400):
        agent.learn(buf, 64)
    last = np.mean([agent.learn(buf, 64) for _ in range(5)])
    assert last < first * 0.2, (first, last)


def test_soft_update_moves_target_towards_online():
    """Verify target network parameters update towards online network with tau > 0.

    Input:  None.
    Output: assertion passes confirming target weights changed after a learning step.
    Flow:   offset online parameters by 1.0, perform 1 learning step with tau=0.5,
            verify target network parameters differ from previous state.
    """
    agent = DQNAgent(make_args(tau=0.5))
    with torch.no_grad():
        for p in agent.online.parameters():
            p.add_(1.0)
    before = [tp.clone() for tp in agent.target.parameters()]
    buf = ReplayBuffer(64)
    for _ in range(64):
        buf.push(np.zeros(4), 0, 1.0, np.zeros(4), 0.0)
    agent.learn(buf, 32)
    moved = any(not torch.equal(b, tp) for b, tp in zip(before, agent.target.parameters()))
    assert moved

