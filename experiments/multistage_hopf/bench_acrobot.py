"""End-to-end bench: 4 arms x 5 reporting seeds on Acrobot.

Pre-registered test from `experiments/mnist_geometric/PRE_REGISTRATION.md`,
amendments 1 and 2. Architectural slots (task=Acrobot, K=2 Hopf stages,
dim-4 eigenspaces only, co-trained linear readout, 84 params, antithetic
NES with pop=40 sigma=0.10 lr=0.05, N=5 reporting seeds, mean reward
over 20 fresh evaluation episodes) are PINNED. Budget per amendment 2:
5e5 env-steps per seed, 8 min wall-clock cap.

The four arms:
    HopfPolicy  + es_train     (the test architecture)
    MLPPolicy   + es_train     (MLP-ES; isolates evolution-vs-gradient)
    MLPPolicy   + ppo_train    (MLP-SGD; gradient baseline)
    MLPPolicy   + random_search (uniform over MLP param space; floor)

For the success criterion (Welch t p<0.05 + 1sigma effect-size vs ALL
baselines), we evaluate each final policy on 20 fresh evaluation
episodes (seeds 1000+seed_offset) and aggregate per-arm.

Outputs go to `experiments/multistage_hopf/checkpoints/results.json`.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

from experiments.multistage_hopf.policies import (
    HopfPolicy,
    MLPPolicy,
)
from experiments.multistage_hopf.trainers import (
    es_train,
    ppo_train,
    random_search,
    evaluate,
    ESConfig,
    PPOConfig,
    RandomSearchConfig,
)


REPORT_SEEDS = (0, 1, 2, 3, 4)
EVAL_SEED_OFFSET = 1_000_000
EVAL_EPISODES = 20

# Budget knobs — set per pre-reg amendment 2.
ENV_STEPS_BUDGET = 500_000
WALL_CLOCK_S = 480.0  # 8 min/seed/arm


def _mk_es_cfg() -> ESConfig:
    # gens chosen to put env-step worst-case roughly at the budget:
    # gens * pop=40 * eps=2 * max=500 = 4e4 * gens. 12 gens -> 4.8e5.
    return ESConfig(
        pop=40, sigma=0.10, lr=0.05,
        generations=15, eval_eps_per_gen=2,
        wall_clock_s=WALL_CLOCK_S,
    )


def _mk_ppo_cfg() -> PPOConfig:
    return PPOConfig(
        lr=3e-3, eps_per_update=4, epochs=4, minibatch=64,
        clip=0.2, gae_lambda=0.95, gamma=0.99,
        entropy_coef=0.01, value_coef=0.5,
        updates=250, wall_clock_s=WALL_CLOCK_S,
    )


def _mk_rs_cfg() -> RandomSearchConfig:
    # ~equal env-step budget: 200 samples * 4 eps * 500 max = 4e5.
    return RandomSearchConfig(
        n_samples=200, sigma=0.5, eval_eps=4, wall_clock_s=WALL_CLOCK_S,
    )


def run_arm(arm_name: str, train_seed: int) -> dict:
    """Train + evaluate one (arm, seed) cell. Returns a result dict."""
    eval_seeds = list(range(EVAL_SEED_OFFSET + train_seed * 1000,
                            EVAL_SEED_OFFSET + train_seed * 1000 + EVAL_EPISODES))

    t0 = time.time()
    if arm_name == "hopf_es":
        policy = HopfPolicy()
        theta, hist = es_train(policy, _mk_es_cfg(), train_seed, log_every=1)
    elif arm_name == "mlp_es":
        policy = MLPPolicy()
        theta, hist = es_train(policy, _mk_es_cfg(), train_seed, log_every=1)
    elif arm_name == "mlp_ppo":
        policy = MLPPolicy()
        theta, hist = ppo_train(policy, _mk_ppo_cfg(), train_seed, log_every=10)
    elif arm_name == "random_search":
        policy = MLPPolicy()
        theta, hist = random_search(policy, _mk_rs_cfg(), train_seed)
    else:
        raise ValueError(arm_name)

    train_time = time.time() - t0

    eval_rewards = []
    for s in eval_seeds:
        from experiments.multistage_hopf.trainers import policy_rollout
        r, _ = policy_rollout(policy, theta, seed=int(s))
        eval_rewards.append(r)
    eval_mean = float(np.mean(eval_rewards))
    eval_std = float(np.std(eval_rewards))

    return {
        "arm": arm_name,
        "train_seed": train_seed,
        "n_params": int(policy.n_params),
        "train_time_s": train_time,
        "history": hist,
        "eval_seeds": eval_seeds,
        "eval_rewards": eval_rewards,
        "eval_mean": eval_mean,
        "eval_std": eval_std,
    }


def welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch's t-test, returns (t, p) two-sided. No scipy dep."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ma = a.mean(); mb = b.mean()
    va = a.var(ddof=1); vb = b.var(ddof=1)
    na = len(a); nb = len(b)
    se = math.sqrt(va / na + vb / nb)
    if se < 1e-12:
        return 0.0, 1.0
    t = (ma - mb) / se
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    )
    # Approximate two-sided p from t and df via the survival function of
    # Student's t. Use a Wallace approximation.
    x = t
    p = _tdist_sf(abs(x), df) * 2.0
    return float(t), float(p)


def _tdist_sf(t: float, df: float) -> float:
    """Survival function of Student's t at t > 0, no scipy."""
    # Use the relation to the regularized incomplete beta function:
    # P(T > t) = 0.5 * I_{df/(df+t^2)}(df/2, 1/2).
    # We implement via Lentz's continued-fraction for the incomplete beta.
    x = df / (df + t * t)
    a = df / 2.0
    b = 0.5
    return 0.5 * _betai(x, a, b)


def _betai(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a,b) via continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Use symmetry to keep x small.
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _betai(1.0 - x, b, a)
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x)) / a
    # Lentz's algorithm for the continued fraction of B(x; a, b)/B(a, b).
    fpmin = 1e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return front * h


def main(out_path: str, arms=None, seeds=REPORT_SEEDS):
    if arms is None:
        arms = ["random_search", "mlp_ppo", "mlp_es", "hopf_es"]
    results = {arm: [] for arm in arms}
    print(f"[bench] arms={arms} seeds={list(seeds)}")
    for arm in arms:
        for s in seeds:
            print(f"[bench] arm={arm} seed={s} starting...")
            t = time.time()
            r = run_arm(arm, s)
            print(f"  [done] eval_mean={r['eval_mean']:.1f} (std {r['eval_std']:.1f}) "
                  f"in {time.time() - t:.1f}s")
            results[arm].append(r)

            # Save incremental results so partial runs are preserved.
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(_summarize(results), f, indent=2)

    summary = _summarize(results)
    print("\n[bench] === Summary ===")
    print(json.dumps(summary["aggregate"], indent=2))
    print("\n[bench] === Welch t-tests (Hopf vs each baseline) ===")
    print(json.dumps(summary["welch_vs_hopf"], indent=2))


def _summarize(results: dict) -> dict:
    aggregate = {}
    by_arm = {}
    for arm, runs in results.items():
        if not runs:
            continue
        means = np.array([r["eval_mean"] for r in runs])
        by_arm[arm] = means
        aggregate[arm] = {
            "n_seeds": int(len(runs)),
            "eval_mean_of_means": float(means.mean()),
            "eval_std_of_means": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
            "per_seed_means": means.tolist(),
            "n_params": runs[0]["n_params"],
        }

    welch = {}
    if "hopf_es" in by_arm:
        h = by_arm["hopf_es"]
        for arm, m in by_arm.items():
            if arm == "hopf_es":
                continue
            t, p = welch_t(h, m)
            effect_sd = (h.mean() - m.mean()) / (m.std(ddof=1) + 1e-12) if len(m) > 1 else 0.0
            welch[f"hopf_vs_{arm}"] = {
                "t": t, "p_two_sided": p,
                "mean_diff": float(h.mean() - m.mean()),
                "effect_size_sd_of_baseline": float(effect_sd),
            }

    return {"aggregate": aggregate, "welch_vs_hopf": welch, "raw": results}


if __name__ == "__main__":
    out = Path("experiments/multistage_hopf/checkpoints/results.json")
    main(str(out))
