# Multi-stage Hopf — does iterated geometric computation add signal?

Self-contained ablation: same input kernel, same readout family,
same evolutionary search, same hyperparameters — only the **number
of Hopf stages K** between the eigenspace projection and the
readout differs.

## Result (Acrobot-v1, 7 seeds × K ∈ {0, 1, 2})

| K | n_params | mean reward | std (across seeds) | mean train time |
|--:|--:|--:|--:|--:|
| 0 | 72 | −213.9 | 17.7 | 240 s |
| 1 | 78 | −220.2 | 13.7 | 352 s |
| 2 | 84 | −218.9 | 21.7 | 473 s |

Welch's t-test, two-sided, K_higher vs K_lower:

| Comparison | t | p | effect size (σ of lower-K) |
|---|---:|---:|---:|
| K=1 vs K=0 | −0.75 | 0.47 | −0.36 |
| K=2 vs K=0 | −0.48 | 0.64 | −0.29 |
| K=2 vs K=1 | +0.13 | 0.90 | +0.09 |

**Adding Hopf stages does not help.** Effect sizes are small and
inconsistent in sign; no comparison comes within an order of
magnitude of significance. The architecture is paying for the
extra rotor parameters and the extra forward-pass cost without
returning anything in fitness.

## What was actually tested

- **Same input kernel** (vMF on the 600-cell, K-independent).
- **Same readout** (linear 24 → 3, no bias, 72 weights, identical
  initialization).
- **Same trainer** (antithetic Gaussian NES, pop = 30, σ = 0.10,
  lr = 0.05, 40 generations, eval_eps = 2).
- **Same task** (Acrobot-v1, 200-step max).
- **Same evaluation** (20 fresh episodes per seed, mean reward).
- **Only K varies** (0, 1, or 2 Hopf stages between projection and
  readout).
- **7 reporting seeds** (0–6).

## What this does and does not say

- It says: at this parameter budget, on this task, the multi-stage
  Hopf chain (rotor → Hopf-project → section-lift → Poincaré warp,
  iterated K times) does not add measurable fitness over a one-shot
  Hopf-projected feature extractor + linear readout.
- It does not say multi-stage Hopf is generally useless. Different
  task, different parameter budget, different optimizer would be
  separate experiments.

## Reproduce

```
PYTHONPATH=. python3 -m pytest experiments/multistage_hopf/tests/test_smoke.py -v
PYTHONPATH=. python3 -m experiments.multistage_hopf.bench_K_sweep
```

Raw per-seed numbers and training histories: `checkpoints/k_sweep.json`.
Bench log with timings: `checkpoints/k_sweep.log`.

The code paths:
- `policies.py` — `HopfPolicy(K=k)` controls the number of stages.
- `bench_K_sweep.py` — head-to-head harness used here.
- `trainers.py` — antithetic NES (`es_train`) shared across K values.
- `acrobot_env.py` — Sutton-spec Acrobot, no gym dependency.
