"""Smoke tests for the multi-stage Hopf bench wiring.

These run each arm for a tiny budget on one seed and assert basic
sanity (right shapes, finite rewards, no NaNs). They are NOT the
pre-registered evaluation; that's `bench_acrobot.py` with the full
budget on reporting seeds.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from experiments.multistage_hopf.acrobot_env import AcrobotEnv, rollout, MAX_STEPS
from experiments.multistage_hopf.policies import (
    HopfPolicy,
    MLPPolicy,
    RandomPolicy,
    acrobot_obs_to_quat,
    vmf_activation,
)
from experiments.multistage_hopf.trainers import (
    es_train,
    sgd_train,
    random_search,
    evaluate,
    ESConfig,
    SGDConfig,
    RandomSearchConfig,
)


def test_acrobot_env_shapes_and_termination():
    env = AcrobotEnv()
    obs = env.reset(seed=0)
    assert obs.shape == (6,)
    assert np.isfinite(obs).all()
    # Random walk, ensure termination by truncation.
    for _ in range(MAX_STEPS):
        obs, r, term, trunc = env.step(np.random.randint(3))
        assert obs.shape == (6,)
        assert np.isfinite(obs).all()
        assert r == -1.0
        if term or trunc:
            break
    assert term or trunc


def test_acrobot_solvable_with_energy_pump():
    """Sanity: a basic energy-pumping heuristic solves Acrobot < 200 steps."""
    def pump(obs):
        return 2 if obs[5] > 0.1 else (0 if obs[5] < -0.1 else 1)
    rs = [rollout(pump, seed=i) for i in range(5)]
    # mean should be much better than -500 (random)
    assert np.mean(rs) > -300, f"Energy pump mean reward {np.mean(rs)} suggests env is broken"


def test_obs_to_quat_normalized():
    rng = np.random.default_rng(0)
    for _ in range(20):
        t1 = rng.uniform(-math.pi, math.pi)
        t2 = rng.uniform(-math.pi, math.pi)
        obs = np.array([math.cos(t1), math.sin(t1), math.cos(t2), math.sin(t2), 0.0, 0.0])
        q = acrobot_obs_to_quat(obs)
        assert q.shape == (4,)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-6


def test_vmf_activation_simplex():
    rng = np.random.default_rng(1)
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    act = vmf_activation(q, kappa=8.0)
    assert act.shape == (120,)
    assert (act >= 0).all()
    assert abs(act.sum() - 1.0) < 1e-9


def test_hopf_policy_param_count():
    hp = HopfPolicy()
    assert hp.n_params == 84, f"Hopf params {hp.n_params} != 84 (pinned)"


def test_mlp_policy_param_count():
    mp = MLPPolicy()
    assert mp.n_params == 83, f"MLP params {mp.n_params} != 83"


def test_hopf_policy_forward_finite():
    hp = HopfPolicy()
    rng = np.random.default_rng(0)
    theta = hp.init_params(rng)
    env = AcrobotEnv()
    obs = env.reset(seed=0)
    for _ in range(50):
        a = hp.act(obs, theta)
        assert a in (0, 1, 2)
        obs, _, term, trunc = env.step(a)
        if term or trunc:
            break


def test_es_runs_one_generation_per_arm():
    """One-generation smoke test for each ES arm; just check the math runs."""
    cfg = ESConfig(pop=4, sigma=0.1, lr=0.05, generations=2, eval_eps_per_gen=1, wall_clock_s=120.0)
    for policy_cls in (HopfPolicy, MLPPolicy):
        policy = policy_cls()
        theta, hist = es_train(policy, cfg, train_seed=0, log_every=1)
        assert theta.shape == (policy.n_params,)
        assert np.isfinite(theta).all()
        assert len(hist) >= 1
        assert hist[-1]["mean"] > -1e6  # finite


def test_sgd_runs_a_few_updates():
    cfg = SGDConfig(lr=0.001, eps_per_update=1, updates=3, wall_clock_s=120.0)
    policy = MLPPolicy()
    theta, hist = sgd_train(policy, cfg, train_seed=0, log_every=1)
    assert theta.shape == (policy.n_params,)
    assert np.isfinite(theta).all()
    assert len(hist) >= 1


def test_random_search_runs():
    cfg = RandomSearchConfig(n_samples=3, sigma=0.5, eval_eps=1, wall_clock_s=60.0)
    policy = MLPPolicy()
    theta, hist = random_search(policy, cfg, train_seed=0)
    assert theta.shape == (policy.n_params,)
    assert hist[0]["best"] > -1e6


def test_evaluate_returns_finite_mean():
    policy = MLPPolicy()
    rng = np.random.default_rng(0)
    theta = policy.init_params(rng)
    mean_r, steps = evaluate(policy, theta, seeds=range(3))
    assert math.isfinite(mean_r)
    assert steps > 0
