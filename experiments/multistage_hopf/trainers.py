"""Training algorithms for the multi-stage Hopf pre-registered test.

Three trainers + one zero-training baseline:

    es_train       — Antithetic Gaussian NES (used for Hopf-ES and MLP-ES)
    sgd_train      — REINFORCE on softmax(logits) for MLP-SGD
    random_search  — uniform sampling over the parameter space
    null           — RandomPolicy needs no training; a tiny stub

All trainers share the same env-step budget (configured externally) and
the same `evaluate(...)` rollout protocol. The reward metric reported
is mean undiscounted episode reward over `eval_episodes` fresh
evaluation episodes at the end of training, with a deterministic seed
offset different from the training seeds.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from experiments.multistage_hopf.acrobot_env import (
    AcrobotEnv,
    rollout,
    MAX_STEPS,
)


# ---------------------------------------------------------------------------
# Common rollout helpers
# ---------------------------------------------------------------------------


def policy_rollout(policy, theta: np.ndarray, seed: int) -> tuple[float, int]:
    """Roll out `policy` with parameters `theta` for one episode at `seed`.

    Returns (total_reward, env_steps).
    """
    env = AcrobotEnv()
    obs = env.reset(seed=seed)
    total = 0.0
    steps = 0
    for _ in range(MAX_STEPS):
        action = int(policy.act(obs, theta))
        obs, r, term, trunc = env.step(action)
        total += r
        steps += 1
        if term or trunc:
            break
    return total, steps


def evaluate(policy, theta: np.ndarray, seeds) -> tuple[float, int]:
    """Mean reward over the given evaluation seeds."""
    rs = []
    total_steps = 0
    for s in seeds:
        r, st = policy_rollout(policy, theta, seed=int(s))
        rs.append(r)
        total_steps += st
    return float(np.mean(rs)), total_steps


# ---------------------------------------------------------------------------
# Antithetic Gaussian NES (Salimans et al. 2017 with antithetic sampling)
# ---------------------------------------------------------------------------


@dataclass
class ESConfig:
    pop: int = 40           # population size (must be even for antithetic)
    sigma: float = 0.10     # exploration noise
    lr: float = 0.05        # gradient step size
    generations: int = 200
    eval_eps_per_gen: int = 4  # episodes per parameter eval
    wall_clock_s: float = 600.0  # 10 minutes


def es_train(policy, cfg: ESConfig, train_seed: int, log_every: int = 25):
    """Antithetic Gaussian NES.

    Returns (theta_final, history) where history is a list of dicts.
    """
    rng = np.random.default_rng(train_seed)
    theta = policy.init_params(rng)
    n = policy.n_params
    half = cfg.pop // 2

    history = []
    t0 = time.time()
    total_steps = 0
    best_eval = -math.inf
    best_theta = theta.copy()

    for gen in range(cfg.generations):
        if time.time() - t0 > cfg.wall_clock_s:
            break

        eps = rng.standard_normal(size=(half, n))
        rewards = np.zeros(cfg.pop)

        for i in range(half):
            theta_p = theta + cfg.sigma * eps[i]
            theta_m = theta - cfg.sigma * eps[i]
            # Per-perturbation seed offset (deterministic per generation).
            base = train_seed * 1_000_000 + gen * 10_000 + i * 10
            r_p = 0.0
            r_m = 0.0
            for j in range(cfg.eval_eps_per_gen):
                rp_, sp = policy_rollout(policy, theta_p, seed=base + j)
                rm_, sm = policy_rollout(policy, theta_m, seed=base + j + 5)
                r_p += rp_
                r_m += rm_
                total_steps += sp + sm
            rewards[2 * i] = r_p / cfg.eval_eps_per_gen
            rewards[2 * i + 1] = r_m / cfg.eval_eps_per_gen

        # Rank-based fitness shaping for stable updates.
        ranks = np.empty_like(rewards)
        ranks[np.argsort(rewards)] = np.arange(cfg.pop)
        shaped = ranks / (cfg.pop - 1) - 0.5  # in [-0.5, 0.5]

        # Antithetic gradient estimate.
        grad = np.zeros(n)
        for i in range(half):
            grad += (shaped[2 * i] - shaped[2 * i + 1]) * eps[i]
        grad /= (cfg.pop * cfg.sigma)

        theta = theta + cfg.lr * grad

        if gen % log_every == 0 or gen == cfg.generations - 1:
            mean_r = float(np.mean(rewards))
            best_r = float(np.max(rewards))
            history.append({
                "gen": gen,
                "mean": mean_r,
                "best": best_r,
                "steps": total_steps,
                "elapsed": time.time() - t0,
            })

        # Track best theta-on-mean. (Cheap; done after gradient step.)
        if rewards.mean() > best_eval:
            best_eval = rewards.mean()
            best_theta = theta.copy()

    return best_theta, history


# ---------------------------------------------------------------------------
# REINFORCE SGD for MLP (gradient through the softmax policy)
# ---------------------------------------------------------------------------


@dataclass
class SGDConfig:
    lr: float = 0.01
    eps_per_update: int = 4   # episodes per gradient step
    updates: int = 2000       # gradient steps; matches ES env-step budget on average
    gamma: float = 1.0        # undiscounted (Acrobot is finite horizon)
    baseline_decay: float = 0.95
    wall_clock_s: float = 600.0


def _mlp_logp_grad(policy, theta: np.ndarray, obs: np.ndarray, action: int):
    """Return (log_prob, grad_log_prob) for an MLP policy under softmax."""
    W1, b1, W2, b2 = policy.split(theta)
    h_pre = W1 @ obs + b1
    h = np.tanh(h_pre)
    z = W2 @ h + b2
    z -= z.max()
    e = np.exp(z)
    p = e / e.sum()
    logp = math.log(max(p[action], 1e-12))

    # dL/dz = p - one_hot(action) for cross-entropy on action.
    dz = p.copy()
    dz[action] -= 1.0
    # We want grad of LOG-prob, which is -dL above (since logp = -CE).
    dz = -dz
    dW2 = np.outer(dz, h)
    db2 = dz
    dh = W2.T @ dz
    dh_pre = dh * (1.0 - h * h)
    dW1 = np.outer(dh_pre, obs)
    db1 = dh_pre
    grad = np.concatenate([dW1.ravel(), db1.ravel(), dW2.ravel(), db2.ravel()])
    return logp, grad


def sgd_train(policy, cfg: SGDConfig, train_seed: int, log_every: int = 100):
    """REINFORCE with a moving-average baseline.

    Acts on MLPPolicy (uses softmax). Hopf is not given an SGD trainer
    because the rotor parameters live on a manifold; per pre-reg, ES
    is the only search method for rotors.
    """
    rng = np.random.default_rng(train_seed)
    theta = rng.standard_normal(policy.n_params) * 0.05
    history = []
    t0 = time.time()
    baseline = -500.0  # Acrobot starts near worst case
    total_steps = 0
    best_eval = -math.inf
    best_theta = theta.copy()

    for upd in range(cfg.updates):
        if time.time() - t0 > cfg.wall_clock_s:
            break

        grad_acc = np.zeros_like(theta)
        rewards = []
        for j in range(cfg.eps_per_update):
            seed_j = train_seed * 1_000_003 + upd * 7 + j
            env = AcrobotEnv()
            obs = env.reset(seed=seed_j)
            traj = []
            for _ in range(MAX_STEPS):
                # Sample action from softmax.
                p = policy.softmax_probs(obs, theta)
                a = int(rng.choice(3, p=p))
                _, grad_logp = _mlp_logp_grad(policy, theta, obs, a)
                obs, r, term, trunc = env.step(a)
                traj.append((grad_logp, r))
                total_steps += 1
                if term or trunc:
                    break
            ep_R = sum(r for (_, r) in traj)
            advantage = ep_R - baseline
            for (g, _) in traj:
                grad_acc += advantage * g
            rewards.append(ep_R)

        grad_acc /= cfg.eps_per_update
        theta = theta + cfg.lr * grad_acc

        baseline = cfg.baseline_decay * baseline + (1.0 - cfg.baseline_decay) * float(np.mean(rewards))

        if upd % log_every == 0 or upd == cfg.updates - 1:
            history.append({
                "update": upd,
                "mean": float(np.mean(rewards)),
                "baseline": baseline,
                "steps": total_steps,
                "elapsed": time.time() - t0,
            })

        if np.mean(rewards) > best_eval:
            best_eval = float(np.mean(rewards))
            best_theta = theta.copy()

    return best_theta, history


# ---------------------------------------------------------------------------
# Random search baseline
# ---------------------------------------------------------------------------


@dataclass
class RandomSearchConfig:
    n_samples: int = 8000  # ~ matches a 200-gen ES at pop=40
    sigma: float = 0.5     # uniform sampling scale
    eval_eps: int = 4
    wall_clock_s: float = 600.0


def random_search(policy, cfg: RandomSearchConfig, train_seed: int):
    rng = np.random.default_rng(train_seed)
    best_theta = rng.normal(scale=cfg.sigma, size=policy.n_params)
    best_r = -math.inf
    t0 = time.time()
    total_steps = 0

    for k in range(cfg.n_samples):
        if time.time() - t0 > cfg.wall_clock_s:
            break
        theta_k = rng.normal(scale=cfg.sigma, size=policy.n_params)
        r_acc = 0.0
        for j in range(cfg.eval_eps):
            r, st = policy_rollout(policy, theta_k, seed=train_seed * 9_999 + k * 11 + j)
            r_acc += r
            total_steps += st
        r_mean = r_acc / cfg.eval_eps
        if r_mean > best_r:
            best_r = r_mean
            best_theta = theta_k.copy()

    history = [{
        "n_samples_used": k + 1,
        "best": best_r,
        "steps": total_steps,
        "elapsed": time.time() - t0,
    }]
    return best_theta, history
