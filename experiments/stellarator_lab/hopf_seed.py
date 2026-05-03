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

CONVENTION (pinned, see CONSULTATION.md Q4):
    omega1 = poloidal winding number    (turns around the core circle)
    omega2 = toroidal winding number    (turns around the symmetry axis)
    iota   = omega1 / omega2            (= poloidal / toroidal)
    linking_number = omega1 * omega2

This matches the standard stellarator convention (iota = poloidal-per-
toroidal). Verification against a live VMEC / DESC import format is the
remaining open piece; the convention itself is now pinned at the
module level via the CONVENTION constant below so any downstream
mismatch will surface as a single point of failure.

Naming: this module's "helicity" is always the volumetric Hopf
invariant H = int A.B dV (units of B^2 . length^4). It is NOT the
QUASR / Landreman 'helicity' field, which is the integer M in {0, 1}
of the quasi-symmetry direction (qs_axis_class). The two are sibling-
but-not-equal topological labels; do not conflate. See
RESEARCH_PROGRAM.md Addendum and CONSULTATION.md Q3.

This module is pure numpy. All gradients are in closed form; no finite
differences, no sympy codegen.
"""
from __future__ import annotations

import numpy as np


CONVENTION = {
    "omega1_role": "poloidal winding number",
    "omega2_role": "toroidal winding number",
    "iota_definition": "omega1 / omega2 (poloidal-per-toroidal)",
    "linking_number_definition": "omega1 * omega2",
    "helicity_meaning": "volumetric Hopf invariant H = int A.B dV "
                       "(NOT the QUASR qs_axis_class)",
}


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

    Exact topological invariant: omega1 * omega2, by the (omega1, omega2)
    cover of the Hopf fibration.
    """
    return omega1 * omega2


def analytic_helicity(omega1, omega2, R=1.0):
    """Magnetic helicity H = int A . B dV in this module's Clebsch-pair
    normalization.

    STATUS: empirically determined closed form, NOT yet derived from
    first principles in this repo. The formula

        H(n, m, R) = 2 * pi^2 * n * m * n! * m! / (n + m)! * R^4

    matches numerical integration to ~5 decimals across 18 (n, m) pairs
    at bbox=10, resolution=128 (see test_analytic_helicity_closed_form
    and test_grid_helicity_matches_closed_form). The fit is to machine-
    precision rationals: H/pi^2 = 2 n m / C(n + m, n) for every
    (n, m) tested with n + m <= 8.

    The C(n + m, n) denominator is consistent with a beta-function
    integral over the Hopf fibres on S^3, but the analytic derivation
    is open (see CONSULTATION.md, Q2). Treat this as a tightly-asserted
    empirical theorem, not a derived one. Do not rely on it for
    out-of-distribution (n, m) pairs without re-running the numerical
    check.

    Symmetric in (n, m) as required by the n <-> m exchange symmetry
    of the underlying Hopfion. R^4 scaling is exact dimensional
    analysis (the only length scale in the construction is R).

    Specific values (R = 1):
        H(1, 1) = pi^2          = 9.8696
        H(2, 1) = 4 pi^2 / 3    = 13.1595
        H(2, 2) = 4 pi^2 / 3    = 13.1595      [note: same as H(2, 1)]
        H(3, 2) = 6 pi^2 / 5    = 11.8435
        H(3, 3) = 9 pi^2 / 10   = 8.8827       [note: SMALLER than H(1, 1)]
        H(4, 4) = 16 pi^2 / 35  = 4.5117       [note: smaller still]

    Important reading: volumetric helicity is NOT the linking number
    squared. The linking number is omega1 * omega2 (so 9 for (3, 3),
    16 for (4, 4)) but H(3, 3) < H(1, 1) and H(4, 4) is half of H(1, 1).
    The combinatorial denominator C(n + m, n) suppresses high-bidegree
    configurations because the Hopf bundle is more tightly compressed
    at higher (n, m): the same R^3 volume hosts a higher-link field at
    smaller per-fibre amplitude, and the total int A.B dV decreases.
    Topological lower bound H <= |link|^2 . H_unit; the closed form
    above gives the actual H, which is much smaller for high (n, m).
    Practical implication: high-(n, m) Hopfions are NOT automatically
    the "best" seeds for downstream optimization (omnigenity or
    otherwise); they carry less volumetric helicity per unit linking.

    Magnetic helicity is gauge-invariant only when B . n_hat vanishes
    on the integration boundary. B decays as |x|^-6 at infinity, so the
    formula above is the limit of integration over all of R^3;
    finite-bounding-box numerical helicity has truncation error scaling
    as bbox^-3.
    """
    import math

    n = int(omega1)
    m = int(omega2)
    return (
        2.0
        * math.pi ** 2
        * n
        * m
        * math.factorial(n)
        * math.factorial(m)
        / math.factorial(n + m)
        * (R ** 4)
    )
