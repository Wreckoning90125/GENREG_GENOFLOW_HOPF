"""
Cartesian-grid sampling and finite-difference witnesses for the Hopf seed.

Layer 1 "minimum bar" verification: given the closed-form B and A from
hopf_seed, sample on a Cartesian box, then check div B = 0 via central
differences, curl A = B via central differences, and integrate
helicity H = int A . B dV by Simpson's rule.

Convergence order is 2 for central differences, so log-log fit of max|div B|
vs grid spacing should give slope approx 2.
"""
from __future__ import annotations

import numpy as np

from hopf_seed import seed_field, seed_vector_potential


def build_grid(bbox, resolution):
    """Uniform Cartesian grid over a box.

    bbox: (xmin, xmax, ymin, ymax, zmin, zmax).
    resolution: int N. Grid has N points per axis.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    z = np.linspace(zmin, zmax, resolution)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    if not (np.isclose(dx, dy) and np.isclose(dy, dz)):
        raise ValueError(
            f"build_grid requires isotropic spacing, got dx={dx}, dy={dy}, dz={dz}"
        )
    return {
        "x": x,
        "y": y,
        "z": z,
        "dx": dx,
        "bbox": bbox,
        "shape": (resolution, resolution, resolution),
    }


def sample_seed_on_grid(grid, omega1, omega2, R=1.0):
    """Evaluate B and A on the grid. Returns arrays shape (3, N, N, N)."""
    X, Y, Z = np.meshgrid(grid["x"], grid["y"], grid["z"], indexing="ij")
    Bx, By, Bz = seed_field(X, Y, Z, omega1, omega2, R)
    Ax, Ay, Az = seed_vector_potential(X, Y, Z, omega1, omega2, R)
    B = np.stack([Bx, By, Bz], axis=0)
    A = np.stack([Ax, Ay, Az], axis=0)
    return B, A


def grid_divergence(F, h):
    """Central-difference divergence of a vector field, shape (3, N, N, N).

    Interior points only; boundary set to NaN so that max-abs calls must
    slice them off (use grid_divergence_interior_max for a clean reduction).
    """
    Fx, Fy, Fz = F[0], F[1], F[2]
    div = np.full_like(Fx, np.nan)
    div[1:-1, :, :] = (Fx[2:, :, :] - Fx[:-2, :, :]) / (2.0 * h)
    tmp = np.full_like(Fy, np.nan)
    tmp[:, 1:-1, :] = (Fy[:, 2:, :] - Fy[:, :-2, :]) / (2.0 * h)
    div = np.where(np.isnan(div) | np.isnan(tmp), np.nan, div + tmp)
    tmp2 = np.full_like(Fz, np.nan)
    tmp2[:, :, 1:-1] = (Fz[:, :, 2:] - Fz[:, :, :-2]) / (2.0 * h)
    div = np.where(np.isnan(div) | np.isnan(tmp2), np.nan, div + tmp2)
    return div


def grid_divergence_interior_max(F, h):
    """Max |div F| over strict interior (drops 1 layer on every face)."""
    Fx, Fy, Fz = F[0], F[1], F[2]
    dFx_dx = (Fx[2:, 1:-1, 1:-1] - Fx[:-2, 1:-1, 1:-1]) / (2.0 * h)
    dFy_dy = (Fy[1:-1, 2:, 1:-1] - Fy[1:-1, :-2, 1:-1]) / (2.0 * h)
    dFz_dz = (Fz[1:-1, 1:-1, 2:] - Fz[1:-1, 1:-1, :-2]) / (2.0 * h)
    div = dFx_dx + dFy_dy + dFz_dz
    return float(np.max(np.abs(div)))


def grid_curl(F, h):
    """Central-difference curl of a vector field, shape (3, N, N, N).

    Returns interior-only array shape (3, N-2, N-2, N-2).
    """
    Fx, Fy, Fz = F[0], F[1], F[2]
    dFz_dy = (Fz[1:-1, 2:, 1:-1] - Fz[1:-1, :-2, 1:-1]) / (2.0 * h)
    dFy_dz = (Fy[1:-1, 1:-1, 2:] - Fy[1:-1, 1:-1, :-2]) / (2.0 * h)
    dFx_dz = (Fx[1:-1, 1:-1, 2:] - Fx[1:-1, 1:-1, :-2]) / (2.0 * h)
    dFz_dx = (Fz[2:, 1:-1, 1:-1] - Fz[:-2, 1:-1, 1:-1]) / (2.0 * h)
    dFy_dx = (Fy[2:, 1:-1, 1:-1] - Fy[:-2, 1:-1, 1:-1]) / (2.0 * h)
    dFx_dy = (Fx[1:-1, 2:, 1:-1] - Fx[1:-1, :-2, 1:-1]) / (2.0 * h)
    curlx = dFz_dy - dFy_dz
    curly = dFx_dz - dFz_dx
    curlz = dFy_dx - dFx_dy
    return np.stack([curlx, curly, curlz], axis=0)


def curl_A_minus_B_max(A, B, h):
    """Max |curl A - B| over the interior (relative to max |B| externally)."""
    curlA = grid_curl(A, h)
    B_int = B[:, 1:-1, 1:-1, 1:-1]
    return float(np.max(np.abs(curlA - B_int)))


def grid_helicity(A, B, h):
    """Helicity H = int A . B dV via composite Simpson's rule in each axis.

    Uses scipy.integrate.simpson; falls back to trapezoid if unavailable.
    """
    dot = A[0] * B[0] + A[1] * B[1] + A[2] * B[2]
    try:
        from scipy.integrate import simpson
        Ix = simpson(dot, dx=h, axis=0)
        Ixy = simpson(Ix, dx=h, axis=0)
        H = simpson(Ixy, dx=h, axis=0)
    except ImportError:
        H = np.trapz(np.trapz(np.trapz(dot, dx=h, axis=0), dx=h, axis=0), dx=h)
    return float(H)


def boundary_flux(B, h):
    """int B . n dA over the box boundary (should be ~0 for decayed B).

    Non-zero values flag that the bbox is too small and helicity is not
    gauge-invariant.
    """
    # Six faces. Assume the box is a cube in the canonical orientation.
    total = 0.0
    # -x, +x faces: normal = -x, +x; integrand = Bx
    total += -np.sum(B[0, 0, :, :]) * h * h
    total += np.sum(B[0, -1, :, :]) * h * h
    # -y, +y
    total += -np.sum(B[1, :, 0, :]) * h * h
    total += np.sum(B[1, :, -1, :]) * h * h
    # -z, +z
    total += -np.sum(B[2, :, :, 0]) * h * h
    total += np.sum(B[2, :, :, -1]) * h * h
    return float(total)


def convergence_study(omega1, omega2, R, bbox, resolutions):
    """Sweep grid resolutions; return a dict of per-resolution residuals.

    For each N in resolutions:
        max|div B|,  max|curl A - B|,  grid helicity.
    The topological ratio grid_helicity / (omega1 * omega2) should be
    roughly resolution-independent and match h_unit_reference() * R^4.
    """
    out = {"resolutions": list(resolutions), "runs": []}
    for N in resolutions:
        grid = build_grid(bbox, N)
        B, A = sample_seed_on_grid(grid, omega1, omega2, R)
        run = {
            "N": int(N),
            "h": float(grid["dx"]),
            "max_abs_divB": grid_divergence_interior_max(B, grid["dx"]),
            "max_abs_curlA_minus_B": curl_A_minus_B_max(A, B, grid["dx"]),
            "helicity": grid_helicity(A, B, grid["dx"]),
            "boundary_flux": boundary_flux(B, grid["dx"]),
        }
        out["runs"].append(run)
    return out
