# Report: Balancing CartPole-v1 with Reinforcement Learning

> Methods: Double DQN (primary) vs REINFORCE (control) vs Random (baseline)
> Budget: 150,000 environment steps, 5 seeds (0-4), evaluation on seeds 100-199
> Numbers sourced from `outputs/results/summary.csv` and `public/docs/tables.md`. No accuracy metric.
>
> Related: [`tables.md`](./tables.md) · [`slides.md`](./slides.md) · Live UI: `public/web/`

---

## 1. Introduction

CartPole-v1 requires an agent to keep a pole balanced on a cart by pushing left or right. This is a classic reinforcement learning task: there is no static dataset, and data are transitions `(s, a, r, s', done)` collected through environment interaction.

Goal: compare Double DQN (value-based, off-policy) and REINFORCE (policy gradient, on-policy) against a random baseline.

---

## 2. Background

### 2.1 Markov Decision Process

The environment is an MDP `(S, A, P, R, γ)`. The agent learns a policy `π(a|s)` that maximizes expected return `G_t = Σ γ^k r_{t+k}`.

### 2.2 Double DQN

Bellman target: `y = r + γ(1 - terminated) · Q_target(s', argmax_a Q_online(s', a))`

Online and target networks reduce overestimation. Soft target update: `θ⁻ ← τθ + (1-τ)θ⁻` with `τ = 0.005`.

### 2.3 REINFORCE with baseline

Discounted return: `G_t = Σ_k γ^k r_{t+k}` computed backward.

Normalize `G` within the update batch as a simple baseline. Loss: `L = -Σ_t log π(a_t|s_t) · G_t - β·H(π)` with entropy bonus `β = 0.005`.

---

## 3. Environment and data

### 3.1 CartPole-v1

| Component | Content |
|---|---|
| State / observation | `x`, `x_dot`, `theta`, `theta_dot` |
| Action | 0 = left, 1 = right |
| Reward | `+1` per non-failed step; return equals balance length |
| Terminated | `\|theta\| > 12°` or `\|x\| > 2.4` m |
| Truncated | 500 steps (CartPole-v1) |
| Reset | Uniform noise in ±0.05 on all four variables |

### 3.2 Observed ranges under random play (1000 steps)

- `x`: [-0.21, 0.30] (declared obs bound ±4.8)
- `theta`: [-0.24, 0.25] (declared obs bound ±0.42 rad)

Figures: `outputs/figures/obs_hist.png`, `outputs/figures/random_hist.png`.

### 3.3 Environment variants

| Variant | Description | Purpose |
|---|---|---|
| V0 | Default, 500-step horizon | Reference evaluation / Solved benchmark |
| V1 | Truncated at 200 steps | Short-horizon training probe |
| V2 | Gaussian observation noise | Robustness |
| V3 | Small reward shaping on theta | Optional training aid; reports still use raw reward |

---

## 4. Methods

### 4.1 Network architectures

- **DQN:** MLP `4 → 128 → 128 → 2` (ReLU), Q-values for two actions.
- **REINFORCE:** MLP `4 → 128 → 2` (logits), Categorical policy.

### 4.2 Main hyperparameters

| Parameter | DQN | REINFORCE |
|---|---|---|
| Learning rate | 1e-3 (Adam) | 3e-3 (Adam) |
| Gamma | 0.99 | 0.99 |
| Hidden units | 128 | 128 |
| Total steps | 150,000 | 150,000 |
| Epsilon | 1.0 → 0.05 (10k steps) | N/A |
| Buffer | 50,000 | N/A (on-policy) |
| Batch | 64 | 1 episode / update |
| Target update | Soft τ=0.005 | N/A |
| Entropy coef | N/A | 0.005 |

### 4.3 Fair comparison rules

- Same 5 training seeds (0-4) for every algorithm.
- Same 150,000 environment-step budget.
- Final evaluation on hold-out seeds 100-199 (never used for checkpoint selection).
- Best checkpoint selected on seeds 1000-1019.
- Reported return always uses raw unshaped reward.

---

## 5. Experimental protocol

- Training seeds: 0, 1, 2, 3, 4
- Checkpoint-selection seeds: 1000-1019
- Final evaluation seeds: 100-199 (100 episodes)
- Error-analysis seeds: 2000-2499 (500 episodes)
- Demo seeds: 3000+

Official Gymnasium Solved benchmark applied: mean return ≥ 475 over 100 test episodes (`env.spec.reward_threshold = 475`).

---

## 6. Results

### 6.1 Before / after training

See [`tables.md`](./tables.md) (Table 1).

- **Random:** mean = 21.54 (expected ~20-25). Floor baseline.
- **DQN base:** after_best mean = **481.50** (4/5 seeds ≥ 475). Solved.
- **REINFORCE base:** after_best mean = **500.00** (5/5 seeds ≥ 475). Solved.

### 6.2 Learning curves

See `outputs/figures/learning_curves.png`.

DQN first reaches eval_mean ≥ 475 near step 35,000, then oscillates (mean drop after peak ~333.58). REINFORCE reaches the threshold near 25,296 steps and stays stable (drop ~8.61).

### 6.3 DQN ablation

See [`tables.md`](./tables.md) (Table 3) and `outputs/figures/ablation_curves.png`.

On CartPole-v1, `nodouble` (499.62) and `hard` (500.00) also solve; Double DQN is not the decisive factor for final score here.

### 6.4 Noise robustness

See `outputs/figures/noise_robustness.png` and [`tables.md`](./tables.md) (Table 4).

REINFORCE remains at 500.0 across tested σ values. DQN degrades clearly for σ > 0.01.

---

## 7. Error analysis

See `outputs/figures/trajectories.png`, `outputs/figures/policy_heatmap.png`, `outputs/results/error_analysis.csv`.

DQN seed-0 best checkpoint: 500/500 analysis episodes end by timeout. No pole-angle or cart-position failures on this checkpoint.

Policy heatmap at `x=0`, `x_dot=0` shows a clear decision boundary in `(theta, theta_dot)`.

---

## 8. Conclusions and limitations

### 8.1 Conclusions

1. Both DQN and REINFORCE far exceed random (~21 → ~480-500).
2. On CartPole-v1, REINFORCE is stronger on final score, stability, and noise robustness.
3. DQN shows post-peak collapse (catastrophic forgetting / policy collapse).
4. Both methods meet the Solved threshold (mean ≥ 475).

### 8.2 Limitations

- CartPole-v1 is simple and fully observed.
- Only 5 seeds; confidence intervals remain wide.
- Strong REINFORCE results may not transfer to harder domains.
- DQN may improve with lower lr, larger buffer, or longer training.

### 8.3 Next steps

- Try PPO / A2C on harder environments.
- Study DQN forgetting in more detail.
- Stress-test REINFORCE under broader noise settings.
- Consider prioritized replay for DQN.

---

## 9. Appendix

### 9.1 Reproduction

See root `README.md`.

### 9.2 Key library versions

- gymnasium: 1.4.0
- torch: 2.14.1
- numpy: 2.5.4
- Python: 3.13.12

### 9.3 Artifacts

- GIFs: `outputs/videos/random.gif`, `dqn_best.gif`, `reinforce_best.gif`, `dqn_new_data.gif`
- Demo notebook: `notebooks/04_demo.ipynb`
- Figures: `outputs/figures/learning_curves.png`, `noise_robustness.png`, `ablation_curves.png`, `trajectories.png`, `policy_heatmap.png`
- Live UI modules: `public/web/index.html`, `public/web/css/styles.css`, `public/web/js/app.js`
- Launch: `python scripts/run_stream.py`
