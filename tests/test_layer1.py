"""
Layer 1 verification suite.

Covers the three deliverables from RESEARCH_PROGRAM.md:
    1. Seeds       -> seed_field / seed_vector_potential, grid sampling
    2. Verification -> 600-cell second witness
    3. Bundle diagnostics -> Berry-phase accumulation

Tolerances are calibrated against empirical smoke-test behaviour of the
current implementation; tighten them only after regressions are confirmed.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest


# -----------------------------------------------------------------------------
# Seed construction
# -----------------------------------------------------------------------------

def test_div_B_converges_second_order_at_1_1():
    """Central-difference max|div B| on the (1,1) grid scales as O(h^2)
    (slope >= 1.8). This is the cleanest case — higher windings have
    sharper spatial features and need finer grids to hit asymptotic order."""
    from hopf_grid import convergence_study

    res = convergence_study(
        1, 1, 1.0,
        bbox=(-3.0, 3.0, -3.0, 3.0, -3.0, 3.0),
        resolutions=[32, 64, 96, 128],
    )
    hs = np.array([r["h"] for r in res["runs"]])
    divs = np.array([r["max_abs_divB"] for r in res["runs"]])
    slope, _ = np.polyfit(np.log(hs), np.log(divs), 1)
    assert slope >= 1.8, f"div B (1,1) slope {slope:.3f} < 1.8"


@pytest.mark.parametrize("omega1,omega2", [(2, 1), (3, 2), (2, 3)])
def test_div_B_monotone_convergent(omega1, omega2):
    """For higher windings, require monotone decrease of max|div B| with
    resolution and at least first-order behaviour (slope >= 1.0)."""
    from hopf_grid import convergence_study

    res = convergence_study(
        omega1, omega2, 1.0,
        bbox=(-3.0, 3.0, -3.0, 3.0, -3.0, 3.0),
        resolutions=[32, 64, 96],
    )
    divs = [r["max_abs_divB"] for r in res["runs"]]
    assert divs[0] > divs[1] > divs[2], (
        f"div B not monotone decreasing: {divs} for ({omega1},{omega2})"
    )
    hs = np.array([r["h"] for r in res["runs"]])
    slope, _ = np.polyfit(np.log(hs), np.log(np.array(divs)), 1)
    assert slope >= 1.5, (
        f"div B slope {slope:.3f} < 1.5 for ({omega1},{omega2})"
    )


def test_curl_A_equals_B():
    """Central-difference curl A - B also converges as O(h^2)."""
    from hopf_grid import convergence_study

    res = convergence_study(
        1, 1, 1.0,
        bbox=(-3.0, 3.0, -3.0, 3.0, -3.0, 3.0),
        resolutions=[32, 64, 96],
    )
    hs = np.array([r["h"] for r in res["runs"]])
    errs = np.array([r["max_abs_curlA_minus_B"] for r in res["runs"]])
    slope, _ = np.polyfit(np.log(hs), np.log(errs), 1)
    assert slope >= 1.7, f"curl A - B slope {slope:.3f} < 1.7"


def test_analytic_invariants():
    from hopf_seed import analytic_iota, analytic_linking_number

    for w1, w2 in [(1, 1), (2, 1), (3, 2), (2, 3), (5, 3)]:
        assert analytic_iota(w1, w2) == pytest.approx(w1 / w2)
        assert analytic_linking_number(w1, w2) == w1 * w2


@pytest.mark.parametrize(
    "n,m,expected_over_pi2",
    [
        (1, 1, 1.0),
        (2, 1, 4.0 / 3.0),
        (1, 2, 4.0 / 3.0),
        (3, 2, 6.0 / 5.0),
        (2, 3, 6.0 / 5.0),
        (3, 3, 9.0 / 10.0),
        (4, 4, 16.0 / 35.0),
    ],
)
def test_analytic_helicity_closed_form(n, m, expected_over_pi2):
    """analytic_helicity matches H/pi^2 = 2nm * n!m! / (n+m)!"""
    import math
    from hopf_seed import analytic_helicity

    H = analytic_helicity(n, m, R=1.0)
    assert H / (math.pi ** 2) == pytest.approx(expected_over_pi2, abs=1e-12)


def test_analytic_helicity_R_scaling():
    """H scales as R^4 by dimensional analysis; the closed form must
    reproduce that exactly."""
    from hopf_seed import analytic_helicity

    H1 = analytic_helicity(2, 3, R=1.0)
    H2 = analytic_helicity(2, 3, R=2.0)
    assert H2 / H1 == pytest.approx(16.0, abs=1e-12)


def test_grid_helicity_matches_closed_form():
    """At large bbox + decent resolution, the numerical grid helicity
    matches the closed-form analytic_helicity to better than 1e-3."""
    from hopf_seed import analytic_helicity
    from hopf_grid import build_grid, sample_seed_on_grid, grid_helicity

    bbox = (-10.0, 10.0, -10.0, 10.0, -10.0, 10.0)
    grid = build_grid(bbox, 96)
    for w1, w2 in [(1, 1), (2, 1), (3, 2)]:
        B, A = sample_seed_on_grid(grid, w1, w2, 1.0)
        H_grid = grid_helicity(A, B, grid["dx"])
        H_an = analytic_helicity(w1, w2, 1.0)
        rel = abs(H_grid - H_an) / abs(H_an)
        assert rel < 1e-3, (
            f"({w1},{w2}): grid H={H_grid}, analytic H={H_an}, rel_err={rel}"
        )


# -----------------------------------------------------------------------------
# 600-cell second witness
# -----------------------------------------------------------------------------

def test_600cell_chain_complex_is_zero():
    """d1 . d0 = 0 and d2 . d1 = 0 structurally on the reduced complex."""
    from hopf_600cell_witness import embed_600cell, check_chain_complex

    emb = embed_600cell(R=1.0)
    dd01, dd12 = check_chain_complex(emb)
    assert dd01 == 0.0
    assert dd12 == 0.0


def test_600cell_witness_runs_end_to_end():
    """The second witness produces finite residuals for (3, 2)."""
    from hopf_seed import seed_field, seed_vector_potential
    from hopf_600cell_witness import run_witness

    def Bfn(x, y, z):
        return seed_field(x, y, z, 3, 2, 1.0)

    def Afn(x, y, z):
        return seed_vector_potential(x, y, z, 3, 2, 1.0)

    r = run_witness(Bfn, Afn, R=1.0)
    assert r["n_vertices"] >= 119
    assert r["chain_d1_d0"] == 0.0
    assert r["chain_d2_d1"] == 0.0
    # Exact + nonexact fractions together should reconstruct the norm
    # within numerical tolerance
    total = r["exact_fraction"] ** 2 + r["nonexact_fraction"] ** 2
    assert 0.99 < total < 1.01, f"Hodge split ||exact||^2+||nonexact||^2 = {total}"
    assert np.isfinite(r["discrete_helicity_crude"])


# -----------------------------------------------------------------------------
# Field-line tracing and iota recovery
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "omega1,omega2,expected_iota,tol",
    [
        (1, 1, 1.0, 0.05),
        (3, 2, 1.5, 0.05),
        (2, 3, 2.0 / 3.0, 0.05),
    ],
)
def test_iota_recovered(omega1, omega2, expected_iota, tol):
    from hopf_seed import seed_field
    from hopf_fieldlines import trace_fieldline, recover_iota

    def Bfn(x, y, z):
        return seed_field(x, y, z, omega1, omega2, 1.0)

    x0 = np.array([1.3, 0.0, 0.25])
    tr = trace_fieldline(Bfn, x0, max_length=80.0, max_steps=8000)
    info = recover_iota(tr["path"], R_core=1.0)
    assert not np.isnan(info["iota"]), f"iota fit failed: {info['message']}"
    # iota may be fit with sign convention -d(theta)/d(phi) depending on
    # field orientation; compare absolute values
    assert abs(abs(info["iota"]) - expected_iota) < tol, (
        f"iota={info['iota']} not within {tol} of {expected_iota}"
    )


# -----------------------------------------------------------------------------
# Berry-phase diagnostic
# -----------------------------------------------------------------------------

def test_berry_routes_agree():
    """Pancharatnam and Clifford-rotor accumulation agree to machine
    precision along a traced field line (tighter than the <1e-8 agreement
    on individual triangles verified in hopf_controller)."""
    from hopf_seed import seed_field
    from hopf_berry_diagnostic import accumulate_along_fieldline

    def Bfn(x, y, z):
        return seed_field(x, y, z, 2, 1, 1.0)

    r = accumulate_along_fieldline(Bfn, np.array([1.3, 0.0, 0.2]), max_length=50.0)
    assert r["routes_agree"] < 1e-10, (
        f"Berry routes diverged: panch={r['pancharatnam_total']} "
        f"cliff={r['clifford_berry_total']} diff={r['routes_agree']}"
    )


def test_berry_phase_at_1_1_is_2pi_multiple():
    """For (1,1) the closed Villarceau field line maps to a great
    circle on S^2; accumulated Pancharatnam phase is a multiple of 2*pi
    to numerical tolerance."""
    from hopf_seed import seed_field
    from hopf_berry_diagnostic import accumulate_along_fieldline

    def B11(x, y, z):
        return seed_field(x, y, z, 1, 1, 1.0)

    # max_length=60 gives ~1 full closure of the Villarceau circle from
    # this start point; the accumulated Pancharatnam phase is 2*pi.
    r = accumulate_along_fieldline(B11, np.array([1.2, 0.0, 0.2]), max_length=60.0)
    phase = r["pancharatnam_total"]
    # Expect a multiple of 2*pi modulo tracer truncation. At max_length=60
    # the empirical phase is exactly 2*pi; tolerate drift of 5e-3.
    residual = abs((phase / (2.0 * np.pi)) - round(phase / (2.0 * np.pi)))
    assert residual < 5e-3, (
        f"(1,1) Berry phase not a 2pi multiple: phase={phase}, "
        f"residual={residual}"
    )


# -----------------------------------------------------------------------------
# I/O round-trip
# -----------------------------------------------------------------------------

def test_hdf5_round_trip():
    from hopf_grid import build_grid, sample_seed_on_grid
    from hopf_io import write_hdf5, read_hdf5

    grid = build_grid((-2.0, 2.0, -2.0, 2.0, -2.0, 2.0), 16)
    B, A = sample_seed_on_grid(grid, 2, 1, 1.0)
    meta = {
        "omega1": 2,
        "omega2": 1,
        "R": 1.0,
        "bbox": list(grid["bbox"]),
        "resolution": 16,
        "iota": 2.0,
        "linking_number": 2,
    }
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "seed.h5")
        write_hdf5(path, grid, B, A, meta)
        g2, B2, A2, m2 = read_hdf5(path)
        assert np.allclose(B, B2)
        assert np.allclose(A, A2)
        assert m2["omega1"] == 2
        assert m2["linking_number"] == 2


def test_vtk_writes_valid_file():
    from hopf_grid import build_grid, sample_seed_on_grid
    from hopf_io import write_vtk

    grid = build_grid((-2.0, 2.0, -2.0, 2.0, -2.0, 2.0), 12)
    B, A = sample_seed_on_grid(grid, 1, 1, 1.0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "seed.vti")
        write_vtk(path, grid, B, A)
        assert os.path.getsize(path) > 1000
        # Re-read sanity check
        import pyvista as pv
        img = pv.read(path)
        assert "B" in img.point_data
        assert img.point_data["B"].shape == (12 * 12 * 12, 3)
