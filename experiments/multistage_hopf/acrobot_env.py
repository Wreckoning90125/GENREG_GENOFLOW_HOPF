"""Acrobot-v1 environment.

Faithful reimplementation of the standard Sutton & Barto / OpenAI Gym
Acrobot-v1 specification, with no gym dependency. The state has 4 d.o.f.
(two link angles + two angular velocities) and the observation is 6-d
(cos/sin of each angle, plus angular velocities). Actions are discrete
torques in {-1, 0, +1} applied to the second joint.

Reward is -1 per step. Episode terminates when the tip of the second
link rises above the pivot by more than one link length, or after
500 steps. Goal is to reach termination as quickly as possible.

Dynamics use the "book" parameters (m1=m2=1, l1=l2=1, lc1=lc2=0.5,
I1=I2=1, g=9.8) and a 4-step RK4 integration with dt=0.05 (i.e. the
control loop runs at 20 Hz, the integrator at 80 Hz).
"""

from __future__ import annotations

import numpy as np


# Book parameters
LINK_LENGTH_1 = 1.0
LINK_LENGTH_2 = 1.0
LINK_MASS_1 = 1.0
LINK_MASS_2 = 1.0
LINK_COM_POS_1 = 0.5
LINK_COM_POS_2 = 0.5
LINK_MOI_1 = 1.0
LINK_MOI_2 = 1.0
GRAVITY = 9.8

DT = 0.2  # control timestep (gym default; integrator subdivides further)
INTEGRATOR_SUBSTEPS = 4
MAX_VEL_1 = 4 * np.pi
MAX_VEL_2 = 9 * np.pi

AVAIL_TORQUE = np.array([-1.0, 0.0, 1.0], dtype=np.float64)

MAX_STEPS = 500


def _dsdt(s: np.ndarray, torque: float) -> np.ndarray:
    """Acrobot equations of motion. s = (t1, t2, dt1, dt2)."""
    m1, m2 = LINK_MASS_1, LINK_MASS_2
    l1 = LINK_LENGTH_1
    lc1, lc2 = LINK_COM_POS_1, LINK_COM_POS_2
    I1, I2 = LINK_MOI_1, LINK_MOI_2
    g = GRAVITY
    t1, t2, dt1, dt2 = s

    d1 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(t2)) + I1 + I2
    d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(t2)) + I2
    phi2 = m2 * lc2 * g * np.cos(t1 + t2 - np.pi / 2.0)
    phi1 = (
        -m2 * l1 * lc2 * dt2 ** 2 * np.sin(t2)
        - 2 * m2 * l1 * lc2 * dt2 * dt1 * np.sin(t2)
        + (m1 * lc1 + m2 * l1) * g * np.cos(t1 - np.pi / 2.0)
        + phi2
    )
    ddt2 = (
        torque + d2 / d1 * phi1 - m2 * l1 * lc2 * dt1 ** 2 * np.sin(t2) - phi2
    ) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
    ddt1 = -(d2 * ddt2 + phi1) / d1
    return np.array([dt1, dt2, ddt1, ddt2], dtype=np.float64)


def _rk4(s: np.ndarray, torque: float, dt: float) -> np.ndarray:
    k1 = _dsdt(s, torque)
    k2 = _dsdt(s + 0.5 * dt * k1, torque)
    k3 = _dsdt(s + 0.5 * dt * k2, torque)
    k4 = _dsdt(s + dt * k3, torque)
    return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def _wrap(x: float, lo: float, hi: float) -> float:
    diff = hi - lo
    while x > hi:
        x -= diff
    while x < lo:
        x += diff
    return x


class AcrobotEnv:
    """Stateful Acrobot-v1 environment.

    Use `reset(seed)` to initialize, `step(action)` to advance, where
    `action ∈ {0, 1, 2}` indexes into AVAIL_TORQUE.
    """

    OBS_DIM = 6
    N_ACTIONS = 3

    def __init__(self) -> None:
        self.state: np.ndarray | None = None
        self.steps = 0
        self._rng: np.random.Generator | None = None

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._rng = np.random.default_rng(seed)
        # Default Acrobot initialization: small uniform noise on each state.
        self.state = self._rng.uniform(low=-0.1, high=0.1, size=4).astype(np.float64)
        self.steps = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        s = self.state
        return np.array(
            [np.cos(s[0]), np.sin(s[0]), np.cos(s[1]), np.sin(s[1]), s[2], s[3]],
            dtype=np.float64,
        )

    def _terminal(self) -> bool:
        s = self.state
        return bool(-np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool]:
        assert self.state is not None, "Call reset() before step()."
        torque = float(AVAIL_TORQUE[int(action)])
        s = self.state
        sub_dt = DT / INTEGRATOR_SUBSTEPS
        for _ in range(INTEGRATOR_SUBSTEPS):
            s = _rk4(s, torque, sub_dt)
        # Wrap angles, clamp velocities (gym convention).
        s[0] = _wrap(s[0], -np.pi, np.pi)
        s[1] = _wrap(s[1], -np.pi, np.pi)
        s[2] = float(np.clip(s[2], -MAX_VEL_1, MAX_VEL_1))
        s[3] = float(np.clip(s[3], -MAX_VEL_2, MAX_VEL_2))
        self.state = s
        self.steps += 1
        terminated = self._terminal()
        truncated = self.steps >= MAX_STEPS
        reward = -1.0
        return self._obs(), reward, terminated, truncated


def rollout(policy_fn, seed: int, max_steps: int = MAX_STEPS) -> float:
    """Run one episode under `policy_fn(obs) -> action_idx` and return total reward."""
    env = AcrobotEnv()
    obs = env.reset(seed=seed)
    total = 0.0
    for _ in range(max_steps):
        action = int(policy_fn(obs))
        obs, r, term, trunc = env.step(action)
        total += r
        if term or trunc:
            break
    return total
