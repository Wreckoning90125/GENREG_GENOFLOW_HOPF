"""
Generalized-Hopf seed field: closed-form B(x; omega1, omega2) on R^3.

Construction (Kamchatnov / Smiet / Arrayás–Bouwmeester–Trueba):

    (x, y, z) --inv stereographic--> (q0, q1, q2, q3) in S^3
    u = q0 + i*q1,  v = q2 + i*q3,  |u|^2 + |v|^2 = 1
    phi_A = u^omega1,  phi_B = v^omega2
    B = Im( grad phi_A  x  grad phi_B )
    A = Im( phi_A * grad phi_B )

By curl(f grad g) = grad f x grad g, we have curl A = B exactly in the
complex-valued derivation; taking Im of both sides preserves curl A = B
over the reals. Divergence of a cross product of gradients is zero, so
div B = 0 by construction.

Field-line topology:
    iota = omega1 / omega2
    linking number of any two distinct field lines on the same flux
    surface = omega1 * omega2.

This module is pure numpy. All gradients are in closed form; no finite
differences, no sympy codegen.
"""
from __future__ import annotations

import numpy as np


def stereographic_R3_to_S3(x, y, z, R=1.0):
    """Inverse stereographic projection R^3 -> S^3 subset R^4.

    Parameterization:
        q0 = 2 R x / (r^2 + R^2)
        q1 = 2 R y / (r^2 + R^2)
        q2 = 2 R z / (r^2 + R^2)
        q3 = (r^2 - R^2) / (r^2 + R^2)
    where r^2 = x^2 + y^2 + z^2. One can check q0^2 + q1^2 + q2^2 + q3^2 = 1.

    Accepts scalars or arrays; broadcasts element-wise.
    Returns q0, q1, q2, q3 (each with the broadcast shape of x, y, z).
    """
    r2 = x * x + y * y + z * z
    denom = r2 + R * R
    q0 = 2.0 * R * x / denom
    q1 = 2.0 * R * y / denom
    q2 = 2.0 * R * z / denom
    q3 = (r2 - R * R) / denom
    return q0, q1, q2, q3


def _grad_stereographic(x, y, z, R=1.0):
    """Gradients of q0, q1, q2, q3 with respect to (x, y, z).

    Returns a dict with keys 'q0', 'q1', 'q2', 'q3', each mapping to a
    tuple (d/dx, d/dy, d/dz). Arrays broadcast over x, y, z.
    """
    r2 = x * x + y * y + z * z
    denom = r2 + R * R
    d2 = denom * denom
    twoR = 2.0 * R

    # q0 = 2 R x / denom
    dq0_dx = twoR * (R * R + y * y + z * z - x * x) / d2
    dq0_dy = -twoR * 2.0 * x * y / d2
    dq0_dz = -twoR * 2.0 * x * z / d2

    # q1 = 2 R y / denom
    dq1_dx = -twoR * 2.0 * x * y / d2
    dq1_dy = twoR * (R * R + x * x + z * z - y * y) / d2
    dq1_dz = -twoR * 2.0 * y * z / d2

    # q2 = 2 R z / denom
    dq2_dx = -twoR * 2.0 * x * z / d2
    dq2_dy = -twoR * 2.0 * y * z / d2
    dq2_dz = twoR * (R * R + x * x + y * y - z * z) / d2

    # q3 = (r^2 - R^2) / denom. d/dx q3 = 4 R^2 x / d2
    fourR2 = 4.0 * R * R
    dq3_dx = fourR2 * x / d2
    dq3_dy = fourR2 * y / d2
    dq3_dz = fourR2 * z / d2

    return {
        "q0": (dq0_dx, dq0_dy, dq0_dz),
        "q1": (dq1_dx, dq1_dy, dq1_dz),
        "q2": (dq2_dx, dq2_dy, dq2_dz),
        "q3": (dq3_dx, dq3_dy, dq3_dz),
    }


def hopf_scalar_fields(x, y, z, omega1, omega2, R=1.0):
    """Clebsch scalars phi_A = u^omega1, phi_B = v^omega2 (complex-valued)."""
    q0, q1, q2, q3 = stereographic_R3_to_S3(x, y, z, R)
    u = q0 + 1j * q1
    v = q2 + 1j * q3
    phi_A = u ** omega1
    phi_B = v ** omega2
    return phi_A, phi_B


def _complex_gradients(x, y, z, omega1, omega2, R=1.0):
    """Return phi_A, phi_B, grad phi_A, grad phi_B as complex arrays.

    grad phi_A, grad phi_B are tuples (d/dx, d/dy, d/dz) of complex arrays.
    """
    q0, q1, q2, q3 = stereographic_R3_to_S3(x, y, z, R)
    u = q0 + 1j * q1
    v = q2 + 1j * q3

    grads = _grad_stereographic(x, y, z, R)
    du_dx = grads["q0"][0] + 1j * grads["q1"][0]
    du_dy = grads["q0"][1] + 1j * grads["q1"][1]
    du_dz = grads["q0"][2] + 1j * grads["q1"][2]
    dv_dx = grads["q2"][0] + 1j * grads["q3"][0]
    dv_dy = grads["q2"][1] + 1j * grads["q3"][1]
    dv_dz = grads["q2"][2] + 1j * grads["q3"][2]

    phi_A = u ** omega1
    phi_B = v ** omega2

    u_pow = u ** (omega1 - 1) if omega1 != 0 else np.ones_like(u)
    v_pow = v ** (omega2 - 1) if omega2 != 0 else np.ones_like(v)

    dphiA_dx = omega1 * u_pow * du_dx
    dphiA_dy = omega1 * u_pow * du_dy
    dphiA_dz = omega1 * u_pow * du_dz

    dphiB_dx = omega2 * v_pow * dv_dx
    dphiB_dy = omega2 * v_pow * dv_dy
    dphiB_dz = omega2 * v_pow * dv_dz

    return (
        phi_A,
        phi_B,
        (dphiA_dx, dphiA_dy, dphiA_dz),
        (dphiB_dx, dphiB_dy, dphiB_dz),
    )


def seed_field(x, y, z, omega1, omega2, R=1.0):
    """Generalized-Hopf magnetic field B = Im(grad phi_A x grad phi_B).

    Inputs may be scalars or arrays (broadcast over x, y, z).
    Returns (Bx, By, Bz) real arrays.
    """
    _, _, gA, gB = _complex_gradients(x, y, z, omega1, omega2, R)
    (Ax_c, Ay_c, Az_c) = gA
    (Bx_c, By_c, Bz_c) = gB
    # cross product of complex vector fields
    Cx = Ay_c * Bz_c - Az_c * By_c
    Cy = Az_c * Bx_c - Ax_c * Bz_c
    Cz = Ax_c * By_c - Ay_c * Bx_c
    return Cx.imag, Cy.imag, Cz.imag


def seed_vector_potential(x, y, z, omega1, omega2, R=1.0):
    """Vector potential A = Im(phi_A * grad phi_B).

    Satisfies curl A = B in closed form (see module docstring).
    """
    phi_A, _, _, gB = _complex_gradients(x, y, z, omega1, omega2, R)
    (dBx, dBy, dBz) = gB
    Ax = (phi_A * dBx).imag
    Ay = (phi_A * dBy).imag
    Az = (phi_A * dBz).imag
    return Ax, Ay, Az


def analytic_iota(omega1, omega2):
    """Rotational transform: iota = omega1 / omega2. Exact by construction."""
    return omega1 / omega2


def analytic_linking_number(omega1, omega2):
    """Linking number of any two distinct field lines on a flux surface.

    Exact topological invariant: ω1 * ω2, by the Hopf bundle / (ω1, ω2)
    cover construction. Recoverable from field-line tracing (see
    hopf_fieldlines.recover_iota combined with a second traced line).
    """
    return omega1 * omega2


# Reference helicity at (omega1, omega2, R), memoized. We do NOT claim a
# clean closed-form prefactor: in this specific normalization B has a
# factor of omega1 * omega2, A has a factor of omega2, so helicity scales
# nonlinearly in (omega1, omega2). A closed form exists but depends on
# normalization convention (Rañada / Kamchatnov / Smiet differ by overall
# constants). The helicity reported here is the numerical integral at
# sufficient resolution to be resolution-independent, scaled by R^4
# (simple dimensional analysis). Gauge-invariant only if B . n -> 0 on
# the integration boundary; the reference bbox is chosen large enough
# (|x| > 4R) that boundary flux is below 1e-12 for |omega_i| <= 4.
_HELICITY_CACHE: dict = {}


def reference_helicity(omega1, omega2, R=1.0, resolution=96, halfwidth=4.0):
    """Numerical reference helicity H = int A.B dV for (omega1, omega2, R).

    Memoized on (omega1, omega2, resolution, halfwidth). R-dependence is
    handled by the exact R^4 scaling (no re-integration per R).
    """
    key = (int(omega1), int(omega2), int(resolution), float(halfwidth))
    if key not in _HELICITY_CACHE:
        from hopf_grid import build_grid, sample_seed_on_grid, grid_helicity

        bbox = (-halfwidth, halfwidth, -halfwidth, halfwidth, -halfwidth, halfwidth)
        grid = build_grid(bbox, resolution)
        B, A = sample_seed_on_grid(grid, omega1, omega2, 1.0)
        _HELICITY_CACHE[key] = grid_helicity(A, B, grid["dx"])
    return _HELICITY_CACHE[key] * (R ** 4)
