"""K=0 vs K=1 vs K=2 head-to-head on Acrobot.

This isolates the *multi-stage Hopf computation* claim: same input
kernel, same readout, same ES, same parameter-counting strategy.
The only difference between arms is the number of Hopf stages K
between the eigenspace projection and the readout.

Why this is a more valid test than the original 4-arm bench:

- No MLP / PPO / random_search baselines that confound architecture
  with optimizer or with parameter-space density. The comparison is
  within the Hopf family.
- Random search is not a tested arm, so the "random dominates 84
  params on Acrobot" failure mode of the original bench is moot.
- Same readout architecture across arms (linear 24->3, 72 weights),
  so the comparison is purely "do K extra Hopf stages add work".
- Single optimizer (antithetic NES) so no SGD-vs-ES confound.
- Bigger ES budget than the original bench so convergence is
  closer; if K=2 has anything to express vs K=0, the budget gives
  it room to.

Three K values:
    K=0: no Hopf stages. The 24-dim feature is the Hopf projection
         of the *raw* eigenspace coefficients. n_params = 72 (just
         the linear readout).
    K=1: one Hopf stage. n_params = 78 (6 rotor + 72 readout).
    K=2: two Hopf stages. n_params = 84 (12 rotor + 72 readout).

Outputs go to `experiments/multistage_hopf/checkpoints/k_sweep.json`.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

from experiments.multistage_hopf.policies import HopfPolicy
from experiments.multistage_hopf.trainers import (
    es_train,
    policy_rollout,
    ESConfig,
)
from experiments.multistage_hopf.bench_acrobot import welch_t


REPORT_SEEDS = (0, 1, 2, 3, 4, 5, 6)
EVAL_SEED_OFFSET = 2_000_000
EVAL_EPISODES = 20
WALL_CLOCK_S = 900.0   # 15 min hard cap; ES typically finishes much earlier


def _mk_es_cfg() -> ESConfig:
    return ESConfig(
        pop=30, sigma=0.10, lr=0.05,
        generations=40, eval_eps_per_gen=2,
        wall_clock_s=WALL_CLOCK_S,
    )


def run_arm(K: int, train_seed: int) -> dict:
    eval_seeds = list(range(EVAL_SEED_OFFSET + train_seed * 1000,
                            EVAL_SEED_OFFSET + train_seed * 1000 + EVAL_EPISODES))
    t0 = time.time()
    policy = HopfPolicy(K=K)
    theta, hist = es_train(policy, _mk_es_cfg(), train_seed, log_every=2)
    train_time = time.time() - t0

    eval_rewards = []
    for s in eval_seeds:
        r, _ = policy_rollout(policy, theta, seed=int(s))
        eval_rewards.append(r)

    return {
        "K": K,
        "train_seed": train_seed,
        "n_params": int(policy.n_params),
        "train_time_s": train_time,
        "history": hist,
        "eval_rewards": eval_rewards,
        "eval_mean": float(np.mean(eval_rewards)),
        "eval_std": float(np.std(eval_rewards)),
        "eval_median": float(np.median(eval_rewards)),
    }


def _summarize(results: dict) -> dict:
    aggregate = {}
    by_K = {}
    for K, runs in results.items():
        if not runs:
            continue
        means = np.array([r["eval_mean"] for r in runs])
        by_K[K] = means
        aggregate[f"K={K}"] = {
            "n_seeds": int(len(runs)),
            "eval_mean_of_means": float(means.mean()),
            "eval_std_of_means": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
            "eval_median_of_means": float(np.median(means)),
            "per_seed_means": means.tolist(),
            "n_params": runs[0]["n_params"],
            "mean_train_time_s": float(np.mean([r["train_time_s"] for r in runs])),
        }

    pairwise = {}
    Ks_sorted = sorted(by_K.keys())
    for i, K_a in enumerate(Ks_sorted):
        for K_b in Ks_sorted[i + 1:]:
            a = by_K[K_a]; b = by_K[K_b]
            t, p = welch_t(b, a)  # Test K_b > K_a
            es = (b.mean() - a.mean()) / (a.std(ddof=1) + 1e-12) if len(a) > 1 else 0.0
            pairwise[f"K={K_b}_vs_K={K_a}"] = {
                "t": t, "p_two_sided": p,
                "mean_diff": float(b.mean() - a.mean()),
                "effect_size_sd_of_lower_K": float(es),
            }

    return {"aggregate": aggregate, "pairwise": pairwise, "raw": results}


def main(out_path: str, K_values=(0, 1, 2), seeds=REPORT_SEEDS):
    results = {K: [] for K in K_values}
    print(f"[bench-K] K_values={K_values} seeds={list(seeds)}")
    for K in K_values:
        for s in seeds:
            print(f"[bench-K] K={K} seed={s} starting...", flush=True)
            t = time.time()
            r = run_arm(K, s)
            print(f"  [done] eval_mean={r['eval_mean']:.1f} (std {r['eval_std']:.1f}, "
                  f"median {r['eval_median']:.1f}) train={r['train_time_s']:.0f}s", flush=True)
            results[K].append(r)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(_summarize(results), f, indent=2)

    summary = _summarize(results)
    print("\n[bench-K] === Aggregate ===")
    print(json.dumps(summary["aggregate"], indent=2))
    print("\n[bench-K] === Pairwise (Welch t, K_higher vs K_lower) ===")
    print(json.dumps(summary["pairwise"], indent=2))


if __name__ == "__main__":
    out = Path("experiments/multistage_hopf/checkpoints/k_sweep.json")
    main(str(out))
