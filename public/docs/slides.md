# Slides: CartPole Reinforcement Learning

> Related: [`report.md`](./report.md) · [`tables.md`](./tables.md) · Live UI: `public/web/`

---

## Slide 1: Title

**Balancing CartPole-v1 with Reinforcement Learning**
Double DQN vs REINFORCE vs Random Policy

---

## Slide 2: Problem and goal

- Cart on a track; pole hinged freely and tends to fall
- Actions: push left (0) or right (1) each step
- Goal: keep the pole upright as long as possible (max 500 steps)
- This is RL, not supervised learning: no labels, no accuracy

---

## Slide 3: Environment: state / action / reward / episode

| | Content |
|---|---|
| **State** | `x`, `x_dot`, `theta`, `theta_dot` |
| **Action** | 0 = left, 1 = right |
| **Reward** | `+1` per step (return = episode length) |
| **Terminated** | `\|theta\| > 12°` or `\|x\| > 2.4` m |
| **Truncated** | 500-step horizon |

---

## Slide 4: Data source: interaction transitions

```
Agent --action--> CartPole-v1
  ^                    |
  +-- (obs, reward, terminated, truncated) <--+
              |
    Transition (s, a, r, s', done)
              |
    DQN: Replay Buffer -> random mini-batches
    REINFORCE: full episode -> discounted returns
```

---

## Slide 5: Random baseline

- mean = **21.54** steps (expected 20-25): environment sanity check
- min = 9, max = 72, std = 10.54
- Floor that learned agents must beat
- GIF: `outputs/videos/random.gif`

---

## Slide 6: Double DQN

**Target (Double DQN):**
```
y = r + γ(1 - terminated) · Q_target(s', argmax_a Q_online(s', a))
```

- Replay buffer 50k, batch 64
- Epsilon-greedy: 1.0 → 0.05 over 10k steps
- Soft target update: τ = 0.005
- MLP: 4 → 128 → 128 → 2 (ReLU)
- Bootstrap only on `terminated`, never on `truncated`

---

## Slide 7: REINFORCE

**Discounted return (backward):**
```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
```

**Loss with normalization and entropy bonus:**
```
L = -Σ_t log π(a_t|s_t) · (G_t - mean(G)) / std(G)  -  β·H(π)
```

- MLP: 4 → 128 → 2 (logits → Categorical)
- lr = 3e-3, entropy coef = 0.005
- On-policy: no replay buffer

---

## Slide 8: Fair experimental setup

| Criterion | Value |
|---|---|
| Training seeds | 0, 1, 2, 3, 4 |
| Budget | 150,000 steps / seed |
| Checkpoint-selection seeds | 1000-1019 |
| Final evaluation seeds | 100-199 (100 episodes) |
| Reporting | mean ± std across 5 seeds |
| Solved benchmark | mean ≥ 475 / 100 episodes |

---

## Slide 9: Learning curves

`outputs/figures/learning_curves.png`

- DQN: eval ≥ 475 near ~35k steps, large post-peak oscillation (drop ~333.58)
- REINFORCE: ≥ 475 near ~25.3k steps, much more stable (drop ~8.61)
- Random: flat around ~21 steps

---

## Slide 10: Before / after table

| Agent | Before | After (best) | Std | Frac 500 | Seeds ≥ 475 |
|---|---|---|---|---|---|
| Random | N/A | 21.54 | 10.54 | 0% | N/A |
| DQN base | 25.57 | **481.50** | 31.31 | 94.6% | 4/5 |
| REINFORCE base | 39.21 | **500.00** | 0.00 | 100% | 5/5 |

---

## Slide 11: Ablation and noise robustness

**DQN ablation:** `outputs/figures/ablation_curves.png`
- Double vs standard DQN
- Soft vs hard target update

**Noise robustness:** `outputs/figures/noise_robustness.png`

| σ | DQN | REINFORCE |
|---|---|---|
| 0.00 | 482.6 | **500.0** |
| 0.01 | 362.6 | **500.0** |
| 0.05 | 184.2 | **500.0** |
| 0.10 | 55.4 | **500.0** |

---

## Slide 12: Error analysis

`outputs/figures/trajectories.png` + `outputs/figures/policy_heatmap.png`

- DQN seed-0 best: 500/500 timeouts (no failure modes observed)
- Policy heatmap: clear boundary in `(theta, theta_dot)`
- Action opposes pole tilt direction

---

## Slide 13: Live demo

- DQN: `outputs/videos/dqn_best.gif`
- REINFORCE: `outputs/videos/reinforce_best.gif`
- New-seed demo: `outputs/videos/dqn_new_data.gif`
- Notebook: `notebooks/04_demo.ipynb`
- Web UI: `public/web/` via `python scripts/run_stream.py`

---

## Slide 14: Conclusions and limitations

**Conclusions:**
1. Both methods far beat random (21 → ~480-500)
2. REINFORCE is more stable and noise-robust than DQN here
3. DQN shows post-peak collapse
4. Both meet Solved (mean ≥ 475)

**Limitations:**
- CartPole is too simple for broad claims
- Only 5 seeds
- Need harder domains (LunarLander, Atari)

---

## Slide 15: Q&A prompts

- Why bootstrap only on `terminated`, not `done`?
- What problems do replay buffer and target networks solve?
- Why is REINFORCE more robust to observation noise here?
- What does the moving average represent, and which window?
- Does reward shaping make evaluation unfair?
