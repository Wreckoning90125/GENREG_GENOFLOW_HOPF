# ================================================================
# Tests for the H^2(G,U(1)) equivariant-ML transfer of
# Warman & Schafer-Nameki (arXiv:2512.13777).
#
# These verify, against the paper's canonical formulas, that:
#   - alpha_N satisfies the 2-cocycle condition and BSW normalization
#   - alpha_N is the non-trivial class in H^2(D_4N,U(1)) = Z_2
#   - beta_N trivializes alpha_N on <r> (paper Eq. A.29)
#   - U_{alpha,beta}(rs,s) reproduces the logical gate T^{1/N} (Table I)
#   - rho_alpha is a genuine projective rep with cocycle alpha_N
#   - the alpha-twisted conv operators are intertwiners and span the
#     full commutant of rho_alpha
#   - the linear- and cocycle-aware hypothesis spaces genuinely differ
#   - the beta-gauge identity D_beta L_alpha D_beta^-1 = beta(k) L(k)
# ================================================================

import cmath
import math

import numpy as np
import pytest

from dihedral_cohomology import (
    Dihedral,
    DihedralCocycle,
    check_beta_trivializes,
    check_cocycle_condition,
    check_normalization,
    is_coboundary,
    reproduce_gate_table,
)
from projective_equivariant import (
    CocycleRepresentation,
    commutant_basis,
    subspace_overlap,
)

NS = [1, 2, 4]


# ---- group axioms ----------------------------------------------
@pytest.mark.parametrize("N", NS)
def test_group_axioms(N):
    G = Dihedral(N)
    idg = G.identity()
    els = G.elements()
    assert len(els) == 8 * N
    for g in els:
        assert G.mul(idg, g) == g
        assert G.mul(g, idg) == g
        assert G.mul(g, G.inverse(g)) == idg
        assert G.mul(G.inverse(g), g) == idg
    # defining relation s r s = r^-1
    sr_s = G.mul(G.mul(G.s, G.r), G.s)
    assert sr_s == G.inverse(G.r)


# ---- cocycle: condition, normalization, non-triviality ---------
@pytest.mark.parametrize("N", NS)
def test_cocycle_condition(N):
    coc = DihedralCocycle(N)
    assert check_cocycle_condition(coc, "raw")
    assert check_cocycle_condition(coc, "alpha")


@pytest.mark.parametrize("N", NS)
def test_normalization(N):
    assert check_normalization(DihedralCocycle(N))


@pytest.mark.parametrize("N", NS)
def test_alpha_is_nontrivial_class(N):
    # alpha_N must NOT be a coboundary (it is the generator of Z_2)
    assert not is_coboundary(DihedralCocycle(N))


# ---- trivialization beta_N -------------------------------------
@pytest.mark.parametrize("N", NS)
def test_beta_trivializes_on_r(N):
    coc = DihedralCocycle(N)
    assert check_beta_trivializes(coc, "r")
    assert check_beta_trivializes(coc, "s")
    assert check_beta_trivializes(coc, "rs")


@pytest.mark.parametrize("N", NS)
def test_beta_matches_paper_formula(N):
    # paper Eq. A.29: beta_N(r) = exp(i pi/(4N))
    coc = DihedralCocycle(N)
    val = coc.ph.to_complex(coc.beta_rot_exp(1))
    assert abs(val - cmath.exp(1j * math.pi / (4 * N))) < 1e-12
    # beta_N(r^{2N}) = +1
    assert coc.ph.is_one(coc.beta_rot_exp(2 * N))


# ---- the canonical gate (Corollary 1 / Table I) ----------------
@pytest.mark.parametrize("N", NS)
def test_reproduces_logical_gate(N):
    tab = reproduce_gate_table(N)
    assert tab["matches"]
    # |1> picks up exactly the T^{1/N} phase
    assert abs(tab["phase1"] - cmath.exp(1j * math.pi / (4 * N))) < 1e-12
    # |0> is untouched
    assert tab["phase0_exp"] == 0


def test_gate_table_clifford_levels():
    # 8N = 2^n  =>  level n;  N=1 -> T (n=3), N=2 -> T^1/2 (n=4) ...
    assert reproduce_gate_table(1)["clifford_level_n"] == 3
    assert reproduce_gate_table(2)["clifford_level_n"] == 4
    assert reproduce_gate_table(4)["clifford_level_n"] == 5


# ---- projective representation ----------------------------------
@pytest.mark.parametrize("N", NS)
def test_rho_alpha_is_projective_rep(N):
    rep = CocycleRepresentation(N, cocycle=True)
    G = rep.G
    for g in G.elements():
        for h in G.elements():
            lhs = rep.L_alpha(g) @ rep.L_alpha(h)
            rhs = rep.coc.alpha(g, h) * rep.L_alpha(G.mul(g, h))
            assert np.max(np.abs(lhs - rhs)) < 1e-12


@pytest.mark.parametrize("N", NS)
def test_twisted_conv_is_intertwiner(N):
    rep = CocycleRepresentation(N, cocycle=True)
    G = rep.G
    for g in G.elements():
        Rg = rep.R_alpha(g)
        for k in (G.r, G.s, G.rs):
            Lk = rep.L_alpha(k)
            assert np.max(np.abs(Lk @ Rg - Rg @ Lk)) < 1e-12


@pytest.mark.parametrize("N", NS)
def test_twisted_conv_spans_commutant(N):
    rep = CocycleRepresentation(N, cocycle=True)
    G = rep.G
    Halpha = rep.conv_basis(twisted=True).reshape(rep.dim, -1).T
    comm = commutant_basis([rep.L_alpha(G.r), rep.L_alpha(G.s)])
    _, _, mean_cos2 = subspace_overlap(Halpha, comm)
    assert mean_cos2 > 1 - 1e-8
    # same number of free parameters: |G|
    assert Halpha.shape[1] == rep.dim


@pytest.mark.parametrize("N", NS)
def test_ordinary_conv_is_intertwiner(N):
    rep = CocycleRepresentation(N, cocycle=True)
    G = rep.G
    for g in G.elements():
        Rg = rep.R(g)
        for k in (G.r, G.s, G.rs):
            Lk = rep.L(k)
            assert np.max(np.abs(Lk @ Rg - Rg @ Lk)) < 1e-12


# ---- the obstruction: the two hypothesis spaces differ ----------
@pytest.mark.parametrize("N", NS)
def test_hypothesis_spaces_differ(N):
    rep = CocycleRepresentation(N, cocycle=True)
    Hlin = rep.conv_basis(twisted=False).reshape(rep.dim, -1).T
    Halpha = rep.conv_basis(twisted=True).reshape(rep.dim, -1).T
    dim_inter, ntot, mean_cos2 = subspace_overlap(Hlin, Halpha)
    # both have |G| parameters but share only the identity direction
    assert ntot == rep.dim
    assert dim_inter == 1
    assert mean_cos2 < 0.5  # genuinely different subspaces


def test_trivial_cocycle_collapses_to_ordinary():
    # with cocycle=None the twisted operators must equal the ordinary
    rep = CocycleRepresentation(2, cocycle=False)
    for g in rep.G.elements():
        assert np.max(np.abs(rep.R_alpha(g) - rep.R(g))) < 1e-12
        assert np.max(np.abs(rep.L_alpha(g) - rep.L(g))) < 1e-12


# ---- beta-gauge recovery ---------------------------------------
@pytest.mark.parametrize("N", NS)
def test_beta_gauge_identity(N):
    # The beta-gauge identity is SUBGROUP-LOCAL: alpha_N only
    # trivializes on K = <r>, so the identity
    #     D_beta @ L_alpha(k) @ D_beta^-1 = beta(k) * L(k)
    # holds on the C[K] block (no global trivialization exists for a
    # non-trivial class -- that is precisely the obstruction).
    rep = CocycleRepresentation(N, cocycle=True)
    coc = DihedralCocycle(N)
    G = rep.G
    Kidx = rep.rot_subspace_indices()
    beta_diag = np.array(
        [coc.ph.to_complex(coc.beta_exp(G.rot(a))) for a in range(G.n_rot)]
    )
    Dbeta = np.diag(beta_diag)
    Dinv = np.diag(1.0 / beta_diag)
    for a in range(G.n_rot):
        k = G.rot(a)
        La_K = rep.L_alpha(k)[np.ix_(Kidx, Kidx)]
        L_K = rep.L(k)[np.ix_(Kidx, Kidx)]
        lhs = Dbeta @ La_K @ Dinv
        bk = coc.ph.to_complex(coc.beta_exp(k))
        assert np.max(np.abs(lhs - bk * L_K)) < 1e-10


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
