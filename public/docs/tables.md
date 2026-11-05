# Experimental Result Tables: CartPole RL

> All numbers come from `outputs/results/summary.csv` and `outputs/results/noise_robustness.csv`. Do not type numbers by hand.
> Std across seeds uses population std (`ddof=0`), matching `libraries/summarize.py`.
>
> Related: [`report.md`](./report.md) · [`slides.md`](./slides.md)

---

## Table 1: Policy comparison before and after training

Evaluation: 100 episodes on test seeds 100-199 (never used for checkpoint selection).

| Agent | Mean return (before) | Mean return (after_best) | Std across seeds | Fraction at 500 | Seeds ≥ 475 |
|---|---|---|---|---|---|
| Random | N/A | 21.54 | 10.54 | 0% | N/A |
| DQN base | 25.57 | **481.50** | 31.31 | 94.6% | 4/5 |
| REINFORCE base | 39.21 | **500.00** | 0.00 | 100% | 5/5 |

---

## Table 2: Sample efficiency and stability

| Agent | Steps to eval_mean ≥ 475 (median) | Late stability std (last 20%) | Drop after peak |
|---|---|---|---|
| DQN base | 35,000 | 129.04 | 333.58 |
| REINFORCE base | 25,296 | 40.04 | 8.61 |

> DQN reaches the threshold earlier in absolute steps but oscillates more after the peak.
> REINFORCE is more stable (drop 8.61 vs 333.58 for DQN).

---

## Table 3: DQN ablation

| Configuration | Mean return (after_best) | Std across seeds | Note |
|---|---|---|---|
| DQN base (Double + Soft) | 481.50 | 31.31 | Primary configuration |
| DQN nodouble | 499.62 | 0.47 | No Double DQN |
| DQN hard update | 500.00 | 0.00 | Double DQN + hard target (tau=1) |

---

## Table 4: Observation-noise robustness (50 episodes / seed)

Models trained on V0, evaluated on V2 for multiple σ. Values are mean ± std across the 5 training seeds.

| σ | DQN base (mean ± std) | REINFORCE base (mean ± std) |
|---|---|---|
| 0.00 | 482.6 ± 34.8 | 500.0 ± 0.0 |
| 0.01 | 362.6 ± 124.0 | 500.0 ± 0.0 |
| 0.05 | 184.2 ± 152.0 | 500.0 ± 0.0 |
| 0.10 | 55.4 ± 37.2 | 500.0 ± 0.0 |

> REINFORCE stays solved under all tested noise levels.
> DQN degrades clearly for σ > 0.01.

---

## Table 5: DQN error analysis (500 episodes, seeds 2000-2499)

Checkpoint: `models/dqn_base_seed0_best.pt`.

| Metric | Value |
|---|---|
| Episodes | 500 |
| Mean length | **500.0** steps |
| End reason `timeout` | 500/500 (100%) |
| Episodes < 200 steps | 0 |

> This checkpoint reaches 500/500 on the analysis set.
> No pole-angle or cart-position failures for this checkpoint.
