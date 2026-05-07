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
# PPO with clipped objective + GAE for the MLP-SGD arm
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    """Per pre-reg amendment 2. Only `lr` is tunable on validation seeds."""
    lr: float = 3e-3
    eps_per_update: int = 4         # rollout episodes per PPO update
    epochs: int = 4                 # SGD passes through the rollout
    minibatch: int = 64
    clip: float = 0.2
    gae_lambda: float = 0.95
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    updates: int = 200
    wall_clock_s: float = 480.0


def _mlp_value_head_init(rng: np.random.Generator, hidden: int) -> np.ndarray:
    """A separate small linear value head on top of the same MLP hidden layer.

    Architecture: shared 6->8 tanh trunk (the existing MLPPolicy);
    actor head 8->3 (already in MLPPolicy params); value head 8->1
    (this function returns its weights).

    The value head is *additional* parameters not counted against
    the MLPPolicy.n_params budget; the pre-reg's parameter accounting
    is over the policy parameters used at action selection time.
    """
    return rng.normal(scale=0.1, size=hidden + 1)  # 8 weights + 1 bias


def _mlp_forward_actor_critic(policy, theta: np.ndarray, value_w: np.ndarray, obs: np.ndarray):
    """Forward pass for actor-critic on the same trunk.

    Returns (probs, value, h) where probs is (3,), value is scalar,
    h is the hidden activation needed for backprop.
    """
    W1, b1, W2, b2 = policy.split(theta)
    h_pre = W1 @ obs + b1
    h = np.tanh(h_pre)
    z = W2 @ h + b2
    z -= z.max()
    e = np.exp(z)
    probs = e / e.sum()
    value = float(value_w[:-1] @ h + value_w[-1])
    return probs, value, h


def _ppo_grads(
    policy, theta: np.ndarray, value_w: np.ndarray,
    obs_batch: np.ndarray, act_batch: np.ndarray,
    old_logp_batch: np.ndarray, adv_batch: np.ndarray,
    ret_batch: np.ndarray,
    clip: float, ent_coef: float, value_coef: float,
):
    """Compute the PPO loss gradient on a minibatch.

    Returns (grad_theta, grad_value_w, info). Hand-coded numpy
    backprop through softmax-actor + linear-value on shared
    tanh trunk.
    """
    n_W1 = policy.n_W1; n_b1 = policy.n_b1; n_W2 = policy.n_W2
    W1, b1, W2, b2 = policy.split(theta)
    H = policy.HIDDEN
    OBS = policy.OBS_DIM

    grad_theta = np.zeros_like(theta)
    grad_v = np.zeros_like(value_w)
    n = len(obs_batch)

    pol_loss = 0.0
    val_loss = 0.0
    ent = 0.0

    for i in range(n):
        obs = obs_batch[i]
        a = int(act_batch[i])
        old_lp = old_logp_batch[i]
        adv = adv_batch[i]
        ret = ret_batch[i]

        # Forward.
        h_pre = W1 @ obs + b1
        h = np.tanh(h_pre)
        z = W2 @ h + b2
        zmax = z.max()
        ez = np.exp(z - zmax)
        probs = ez / ez.sum()
        value = float(value_w[:-1] @ h + value_w[-1])
        logp_a = math.log(max(probs[a], 1e-12))

        # PPO clipped policy loss.
        ratio = math.exp(logp_a - old_lp)
        clipped = max(min(ratio, 1.0 + clip), 1.0 - clip)
        if (ratio - 1.0) * adv >= (clipped - 1.0) * adv:
            # min(ratio*adv, clipped*adv) = clipped*adv
            unclipped = False
        else:
            unclipped = True
        # If unclipped path is chosen, gradient flows through ratio; else 0.
        # Loss = -min(ratio*adv, clipped*adv); we backprop -unclipped*adv as
        # the upstream gradient w.r.t. logp_a (since dratio/dlogp_a = ratio).
        if unclipped:
            d_logp = -ratio * adv  # d(-ratio*adv)/d(logp_a)
        else:
            # The clipped branch does not depend on logp_a -> no gradient.
            d_logp = 0.0

        # Entropy bonus: H = -sum p log p; grad_z H = p * (logp - sum p logp)
        # = p * (logp + H_val). We add ent_coef * H -> d_loss/dlogp += -ent_coef * grad
        # Simpler: grad of entropy wrt z.
        H_val = -float((probs * np.log(probs + 1e-12)).sum())
        ent += H_val

        # ---- Backprop policy ----
        # logp_a = z[a] - logsumexp(z). d logp_a / d z = -p, with +1 at a.
        dz_pol = -probs.copy()
        dz_pol[a] += 1.0
        dz_pol *= d_logp  # chain through d_logp
        # Entropy grad wrt z: d_entropy/dz_k = -p_k * (1 + log p_k) + p_k * sum p_j (1 + log p_j)
        # = -p_k * log p_k + p_k * sum p_j log p_j
        # = p_k * (sum_j p_j log p_j - log p_k)
        # = -p_k * (log p_k - sum p_j log p_j)
        # = -p_k * (log p_k + H_val)
        dz_ent = -probs * (np.log(probs + 1e-12) + H_val)
        # We want to MAXIMIZE entropy, so subtract ent_coef * H from loss:
        # contribution to dz: -ent_coef * dz_ent.
        dz = dz_pol - ent_coef * dz_ent

        # Value loss: 0.5 * (value - ret)^2; grad wrt value = (value - ret).
        v_err = value - ret
        val_loss += 0.5 * v_err * v_err
        d_val = value_coef * v_err

        # Backprop dz through W2, b2.
        dW2 = np.outer(dz, h)
        db2 = dz
        # Backprop value through value_w (linear).
        dvalue_w = np.empty_like(value_w)
        dvalue_w[:-1] = d_val * h
        dvalue_w[-1] = d_val

        # Backprop into h.
        dh = W2.T @ dz + d_val * value_w[:-1]
        dh_pre = dh * (1.0 - h * h)

        dW1 = np.outer(dh_pre, obs)
        db1 = dh_pre

        # Pack into flat grad_theta.
        i0 = 0
        grad_theta[i0 : i0 + n_W1] += dW1.ravel(); i0 += n_W1
        grad_theta[i0 : i0 + policy.n_b1] += db1.ravel(); i0 += policy.n_b1
        grad_theta[i0 : i0 + n_W2] += dW2.ravel(); i0 += n_W2
        grad_theta[i0 : i0 + policy.n_b2] += db2.ravel()

        grad_v += dvalue_w

        pol_loss += -min(ratio * adv, clipped * adv)

    grad_theta /= n
    grad_v /= n

    info = {
        "pol_loss": pol_loss / n,
        "val_loss": val_loss / n,
        "entropy": ent / n,
    }
    return grad_theta, grad_v, info


def ppo_train(policy, cfg: PPOConfig, train_seed: int, log_every: int = 10):
    """PPO with clipped objective + GAE on a 6 -> 8 -> 3 MLP.

    Adds a small linear value head (8 -> 1) on top of the same hidden
    layer; value-head parameters are NOT counted against
    `policy.n_params` for the parameter-budget comparison (the policy
    is what selects actions at evaluation time).
    """
    rng = np.random.default_rng(train_seed)
    theta = rng.standard_normal(policy.n_params) * 0.1
    value_w = _mlp_value_head_init(rng, policy.HIDDEN)
    history = []
    t0 = time.time()
    total_steps = 0
    best_eval = -math.inf
    best_theta = theta.copy()

    for upd in range(cfg.updates):
        if time.time() - t0 > cfg.wall_clock_s:
            break

        # Roll out cfg.eps_per_update episodes.
        all_obs = []
        all_act = []
        all_logp = []
        all_rew = []
        all_val = []
        all_done = []
        ep_returns = []

        for j in range(cfg.eps_per_update):
            env = AcrobotEnv()
            seed_j = train_seed * 1_000_003 + upd * 7 + j
            obs = env.reset(seed=seed_j)
            ep_R = 0.0
            ep_obs = []
            ep_act = []
            ep_logp = []
            ep_rew = []
            ep_val = []
            for _ in range(MAX_STEPS):
                probs, val, _ = _mlp_forward_actor_critic(policy, theta, value_w, obs)
                a = int(rng.choice(3, p=probs))
                lp = math.log(max(probs[a], 1e-12))
                ep_obs.append(obs.copy())
                ep_act.append(a)
                ep_logp.append(lp)
                ep_val.append(val)
                obs, r, term, trunc = env.step(a)
                ep_rew.append(r)
                ep_R += r
                total_steps += 1
                if term or trunc:
                    break
            # Compute GAE and returns.
            T = len(ep_rew)
            advs = np.zeros(T)
            gae = 0.0
            next_val = 0.0  # bootstrap with 0 since terminal/truncated
            for t in reversed(range(T)):
                delta = ep_rew[t] + cfg.gamma * next_val - ep_val[t]
                gae = delta + cfg.gamma * cfg.gae_lambda * gae
                advs[t] = gae
                next_val = ep_val[t]
            rets = advs + np.array(ep_val)

            all_obs.append(np.array(ep_obs))
            all_act.append(np.array(ep_act))
            all_logp.append(np.array(ep_logp))
            all_rew.append(np.array(ep_rew))
            all_val.append(np.array(ep_val))
            all_done.append(advs)  # reused as adv buffer per-episode
            ep_returns.append(ep_R)

        obs_batch = np.concatenate(all_obs, axis=0)
        act_batch = np.concatenate(all_act, axis=0)
        old_logp_batch = np.concatenate(all_logp, axis=0)
        adv_batch = np.concatenate([np.asarray(a) for a in all_done], axis=0)
        # Compute rets per episode:
        ret_batch = np.concatenate([np.asarray(all_done[k]) + np.asarray(all_val[k]) for k in range(len(all_val))], axis=0)
        # Normalize advantages.
        if adv_batch.std() > 1e-8:
            adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

        # PPO update epochs.
        N = len(obs_batch)
        for ep in range(cfg.epochs):
            idx = rng.permutation(N)
            for start in range(0, N, cfg.minibatch):
                end = start + cfg.minibatch
                mb = idx[start:end]
                gth, gv, info = _ppo_grads(
                    policy, theta, value_w,
                    obs_batch[mb], act_batch[mb],
                    old_logp_batch[mb], adv_batch[mb], ret_batch[mb],
                    cfg.clip, cfg.entropy_coef, cfg.value_coef,
                )
                # Gradient descent on loss => -lr * grad.
                theta = theta - cfg.lr * gth
                value_w = value_w - cfg.lr * gv

        mean_R = float(np.mean(ep_returns))
        if mean_R > best_eval:
            best_eval = mean_R
            best_theta = theta.copy()

        if upd % log_every == 0 or upd == cfg.updates - 1:
            history.append({
                "update": upd,
                "mean": mean_R,
                "steps": total_steps,
                "elapsed": time.time() - t0,
            })

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
