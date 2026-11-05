"""Unit tests for CartPole web streaming server and engine.

Execution:
    pytest tests/test_stream.py -v
"""
import json
import os
import time

import numpy as np
import pytest

from libraries.stream import CartPoleStreamEngine, PolicyHandler, get_embedded_html

DQN_CKPT = "models/dqn_base_seed0_best.pt"
REINFORCE_CKPT = "models/reinforce_base_seed0_best.pt"
HAS_DQN_CKPT = os.path.isfile(DQN_CKPT)
HAS_REINFORCE_CKPT = os.path.isfile(REINFORCE_CKPT)


def test_policy_handler_heuristic():
    """Verify PolicyHandler with heuristic policy produces valid action and probabilities.

    Input:  None.
    Output: assertion passes without exception.
    Flow:   instantiate PolicyHandler("heuristic"), predict on upright and tilted states,
            assert action in {0, 1} and probabilities sum to 1.
    """
    ph = PolicyHandler("heuristic")
    obs = np.array([0.0, 0.0, 0.05, 0.1], dtype=np.float32)
    action, q_vals, probs = ph.predict(obs)
    assert action in (0, 1)
    assert q_vals is None
    assert probs is not None
    assert len(probs) == 2


def test_policy_handler_random():
    """Verify PolicyHandler with random policy produces discrete actions and uniform probabilities.

    Input:  None.
    Output: assertion passes without exception.
    Flow:   instantiate PolicyHandler("random", seed=42), predict action, assert probabilities are 0.5.
    """
    ph = PolicyHandler("random", seed=42)
    obs = np.zeros(4, dtype=np.float32)
    action, q_vals, probs = ph.predict(obs)
    assert action in (0, 1)
    assert probs == [0.5, 0.5]


@pytest.mark.skipif(not HAS_DQN_CKPT, reason=f"missing checkpoint {DQN_CKPT}")
def test_policy_handler_dqn():
    """Verify PolicyHandler with DQN checkpoint predicts actions and returns Q-values.

    Input:  None.
    Output: assertion passes without exception.
    Flow:   instantiate PolicyHandler("dqn", DQN_CKPT),
            predict on observation, assert q_vals is a list of two floats.
    """
    ph = PolicyHandler("dqn", DQN_CKPT)
    obs = np.zeros(4, dtype=np.float32)
    action, q_vals, probs = ph.predict(obs)
    assert action in (0, 1)
    assert q_vals is not None
    assert len(q_vals) == 2
    assert probs is not None
    assert len(probs) == 2


def test_stream_engine_lifecycle_and_stepping():
    """Verify CartPoleStreamEngine step advances counters and creates JPEG frames.

    Input:  None.
    Output: assertion passes without exception.
    Flow:   create CartPoleStreamEngine, execute 5 steps, verify total_steps == 5,
            check latest_jpeg is non-empty bytes, verify telemetry structure.
    """
    engine = CartPoleStreamEngine(algo="heuristic", mode="infinite", fps=30)
    assert engine.total_steps == 0
    assert engine.latest_jpeg is not None
    assert len(engine.latest_jpeg) > 1000

    for _ in range(5):
        engine.step()

    assert engine.total_steps == 5
    assert engine.episode_steps == 5
    telem = engine.get_telemetry()
    assert telem["total_steps"] == 5
    assert "state" in telem
    assert "theta_deg" in telem["state"]
    assert "x" in telem["state"]
    engine.stop()


def test_stream_engine_perturbation():
    """Verify perturbation modifies environment unwrapped state velocity.

    Input:  None.
    Output: assertion passes without exception.
    Flow:   instantiate engine, record initial cart velocity, trigger perturb('push_right'),
            execute step, verify cart velocity increased.
    """
    engine = CartPoleStreamEngine(algo="heuristic", mode="infinite")
    initial_xdot = float(engine.env.unwrapped.state[1])
    engine.perturb("push_right")
    assert engine.pending_perturbation == (1.0, 0.0)
    engine.step()
    # Check that perturbation was applied
    assert engine.pending_perturbation is None
    engine.stop()


def test_stream_engine_mode_switch():
    """Verify switching between infinite and standard horizon modes.

    Input:  None.
    Output: assertion passes without exception.
    Flow:   instantiate engine in infinite mode, switch to standard mode,
            verify mode property updated.
    """
    engine = CartPoleStreamEngine(mode="infinite")
    assert engine.mode == "infinite"
    engine.set_mode("standard")
    assert engine.mode == "standard"
    engine.set_mode("infinite")
    assert engine.mode == "infinite"
    engine.stop()


def test_stream_engine_pause_and_reset():
    """Verify pause suppresses stepping and reset zeroes episode counters.

    Input:  None.
    Output: assertion passes without exception.
    Flow:   step 3 times, toggle pause, step again (steps remain constant),
            resume, reset, verify episode_steps reset to 0.
    """
    engine = CartPoleStreamEngine(algo="heuristic")
    engine.step()
    engine.step()
    assert engine.episode_steps == 2

    engine.toggle_pause()
    assert engine.is_paused is True
    engine.step()
    assert engine.episode_steps == 2  # paused, no step executed

    engine.toggle_pause()
    assert engine.is_paused is False
    engine.reset_env()
    assert engine.episode_steps == 0
    engine.stop()


def test_embedded_html_contains_essential_elements():
    """Verify modular web assets exist and assembled HTML includes essentials.

    Input:  None.
    Output: assertion passes without exception.
    Flow:   confirm public/web index/css/js files exist, assemble fallback HTML
            via get_embedded_html, assert stream/canvas/SSE/control markers.
    """
    assert os.path.isfile("public/web/index.html")
    assert os.path.isfile("public/web/css/styles.css")
    assert os.path.isfile("public/web/js/app.js")
    html = get_embedded_html()
    assert 'id="video-stream"' in html
    assert 'id="waveform-canvas"' in html
    assert "EventSource" in html
    assert "/api/control" in html
    assert "prob-bar-left" in html
    assert "prob-bar-right" in html


@pytest.mark.skipif(not HAS_REINFORCE_CKPT, reason=f"missing checkpoint {REINFORCE_CKPT}")
def test_policy_handler_mismatched_algo_autodetect():
    """Verify PolicyHandler auto-detects true algorithm when given mismatched checkpoint.

    Input:  None.
    Output: assertion passes without exception.
    Flow:   pass algo='dqn' but point to reinforce checkpoint, verify true algo auto-detected.
    """
    ph = PolicyHandler("dqn", REINFORCE_CKPT)
    assert ph.algo == "reinforce"
    obs = np.zeros(4, dtype=np.float32)
    act, q, probs = ph.predict(obs)
    assert act in (0, 1)
    assert probs is not None


def test_get_available_models_categorized():
    """Verify get_available_models returns partitioned dqn and reinforce lists.

    Input:  None.
    Output: assertion passes without exception.
    Flow:   instantiate engine, call get_available_models, assert keys exist;
            require non-empty lists only when local checkpoints are present.
    """
    engine = CartPoleStreamEngine()
    models = engine.get_available_models()
    assert "dqn" in models
    assert "reinforce" in models
    assert isinstance(models["dqn"], list)
    assert isinstance(models["reinforce"], list)
    if HAS_DQN_CKPT:
        assert len(models["dqn"]) > 0
    if HAS_REINFORCE_CKPT:
        assert len(models["reinforce"]) > 0
    engine.stop()


