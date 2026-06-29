# ================================================================
# Group 2-cocycles for the dihedral family D_{4N}, transferred
# from Warman & Schafer-Nameki, "Transversal Clifford-Hierarchy
# Gates via Non-Abelian Surface Codes" (arXiv:2512.13777v1).
#
# This module re-implements, verifies, and exposes the paper's
# *algebraic* machinery so it can be reused as an ML substrate:
#
#   alpha_N  : the non-trivial class in H^2(D_{4N}, U(1)) = Z_2
#              built from the central extension D_{8N} -> D_{4N}
#              (paper Eq. A.20-A.23).
#   beta_N   : the 1-cochain that *trivializes* alpha_N on the
#              cyclic subgroup <r> = Z_{4N}   (paper Eq. A.29).
#   U_alpha_beta : the transversal phase (paper Eq. II.22)
#              U(g1,g2) = alpha(g1,g2) b3(g1 g2)/(b1(g1) b2(g2)).
#
# We work in EXACT integer-exponent arithmetic: every U(1) phase
# is stored as an integer e in Z/(8N) standing for exp(i*pi*e/(4N))
# = exp(2*pi*i * e/(8N)). Phase multiplication is integer addition
# mod 8N, so the cocycle identity, the normalization conditions of
# Beigi-Shor-Whalen (paper Eq. A.18) and the trivialization
# alpha|<r> = delta beta are all checked with zero floating error.
#
# The headline canonical check (reproduce_gate_table) recovers the
# paper's Table I: the SPT-stacking automorphism realizes the
# logical phase gate T^{1/N} = diag(1, exp(i*pi/(4N))).
#
# Nothing here is learned. Everything is fixed by the cohomology of
# D_{4N}, exactly as the paper's gate is fixed by representation
# theory -- this is the "build the symmetry structure, verify
# against canonical, then use it" pattern of the repo.
# ================================================================

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass


# ----------------------------------------------------------------
# The dihedral group D_{4N} = <r, s | r^{4N} = s^2 = 1, s r s = r^-1>
# of order 8N (paper Eq. I.7 / III.21).  An element r^a s^j is the
# pair (a, j) with a in {0,..,4N-1}, j in {0,1}.
# ----------------------------------------------------------------
@dataclass(frozen=True)
class Dihedral:
    """The dihedral group D_{4N} of order 8N (paper's convention)."""

    N: int

    @property
    def n_rot(self) -> int:
        """Order of the rotation subgroup <r> = Z_{4N}."""
        return 4 * self.N

    @property
    def order(self) -> int:
        return 8 * self.N

    def elements(self):
        """All 8N elements as (a, j) pairs, r^a s^j."""
        return [(a, j) for j in (0, 1) for a in range(self.n_rot)]

    def identity(self):
        return (0, 0)

    def mul(self, g, h):
        """Group multiplication: (r^a s^j)(r^b s^k).

        Using s r s = r^-1, i.e. s^j r^b = r^{(1-2j) b} s^j, so
        (r^a s^j)(r^b s^k) = r^{a + (1-2j) b} s^{j+k}   (paper A.22).
        """
        a, j = g
        b, k = h
        a2 = (a + (1 - 2 * j) * b) % self.n_rot
        j2 = (j + k) % 2
        return (a2, j2)

    def inverse(self, g):
        a, j = g
        if j == 0:
            return ((-a) % self.n_rot, 0)
        # reflections are involutions: (r^a s)^2 = id
        return (a, 1)

    def index(self, g):
        """Stable index of g in self.elements() (rotations first)."""
        a, j = g
        return j * self.n_rot + a

    # convenience element accessors -------------------------------
    @property
    def r(self):
        return (1, 0)

    @property
    def s(self):
        return (0, 1)

    @property
    def rs(self):
        return (1, 1)

    def rot(self, a):
        return (a % self.n_rot, 0)


# ----------------------------------------------------------------
# Exact U(1) phases as integer exponents in Z/(8N).
#   exponent e  <->  exp(i * pi * e / (4N)) = exp(2 pi i * e / (8N))
# +1 -> 0 ,  -1 -> 4N ,  i -> 2N ,  -i -> 6N
# ----------------------------------------------------------------
class PhaseRing:
    """Exact arithmetic in the group of 8N-th roots of unity."""

    def __init__(self, N: int):
        self.N = N
        self.modulus = 8 * N  # full circle in exponent units

    def to_complex(self, e: int) -> complex:
        return cmath.exp(1j * math.pi * (e % self.modulus) / (4 * self.N))

    def mul(self, e1: int, e2: int) -> int:
        return (e1 + e2) % self.modulus

    def inv(self, e: int) -> int:
        return (-e) % self.modulus

    def is_one(self, e: int) -> bool:
        return e % self.modulus == 0


# ----------------------------------------------------------------
# The 2-cocycle alpha_N in H^2(D_{4N}, U(1)) = Z_2.
#
# Built exactly as the paper does (App. A.3):
#   * central extension D_{8N} -> D_{4N}, kernel <r^{4N}> = Z_2,
#     lift l(r^a s^j) = r^a s^j  (same a in 0..4N-1, j in 0,1);
#   * phi(g,h) = r^{4N} iff the D_{8N} product carries the exponent
#     past 4N, with lambda(r^{4N}) = -1, giving the raw cocycle
#     alpha'_N (paper Eq. A.23);
#   * a 1-cochain kappa_N (Eq. A.26) renormalizes alpha'_N to
#     alpha_N (Eq. A.27) obeying the Beigi-Shor-Whalen normalization
#     (Eq. A.18) required for compatibility with the lattice vertex
#     operators.
# ----------------------------------------------------------------
class DihedralCocycle:
    """alpha_N and its trivialization beta_N for D_{4N}."""

    def __init__(self, N: int):
        self.N = N
        self.G = Dihedral(N)
        self.ph = PhaseRing(N)

    # ---- raw cocycle from the central extension (Eq. A.23) ------
    def alpha_raw_exp(self, g, h) -> int:
        """alpha'_N(g,h) as a phase exponent: 0 (+1) or 4N (-1)."""
        a, j = g
        b, k = h
        # exponent of r in the D_{8N} product, taken mod 8N
        e = (a + (1 - 2 * j) * b) % (8 * self.N)
        # the kernel generator r^{4N} (-> lambda = -1) appears iff
        # the un-reduced rotation exponent reaches 4N
        return 4 * self.N if e >= 4 * self.N else 0

    # ---- normalization 1-cochain kappa_N (Eq. A.26) ------------
    def kappa_exp(self, g) -> int:
        """kappa_N(r^a s^j) as a phase exponent.

        Generalized from the paper's D_4 statement: the special
        rotation is r^{2N} (the order-2 central rotation), with
        kappa = -i there, kappa = -1 above it, +1 otherwise; only
        on the pure-rotation sector j = 0.
        """
        a, j = g
        if j == 0 and a == 2 * self.N:
            return 6 * self.N  # -i
        if j == 0 and a > 2 * self.N:
            return 4 * self.N  # -1
        return 0  # +1

    def _delta_kappa_exp(self, g, h) -> int:
        """(delta kappa)(g,h) = kappa(g) kappa(h) / kappa(gh)."""
        gh = self.G.mul(g, h)
        return (self.kappa_exp(g) + self.kappa_exp(h) - self.kappa_exp(gh)) % (8 * self.N)

    # ---- normalized cocycle alpha_N (Eq. A.27) -----------------
    def alpha_exp(self, g, h) -> int:
        """alpha_N(g,h) = alpha'_N(g,h) * (delta kappa)(g,h), as exponent."""
        return (self.alpha_raw_exp(g, h) + self._delta_kappa_exp(g, h)) % (8 * self.N)

    def alpha(self, g, h) -> complex:
        return self.ph.to_complex(self.alpha_exp(g, h))

    # ---- raw trivialization beta'_N on <r> (Eq. A.25) ----------
    def beta_raw_rot_exp(self, a: int) -> int:
        """beta'_N(r^a) = exp(i pi a/(4N))  ->  exponent a (Eq. A.25).

        This trivializes the *un-normalized* cocycle alpha'_N on <r>
        (before the kappa_N renormalization), i.e. alpha'_N|<r> = delta beta'.
        """
        return a % (8 * self.N)

    # ---- trivialization beta_N on <r> (Eq. A.29) ---------------
    def beta_rot_exp(self, a: int) -> int:
        """beta_N(r^a) as a phase exponent (a in 0..4N-1).

        beta_N(r^a) = + exp(i pi a/(4N))   if a < 2N
                      + 1                   if a = 2N
                      - exp(i pi a/(4N))    if a > 2N
        In exponent units that is a, 0(==2N->2N? see below), and
        a + 4N respectively; we return the reduced exponent.
        """
        a = a % (4 * self.N)
        if a < 2 * self.N:
            return a % (8 * self.N)
        if a == 2 * self.N:
            return 0  # +1 (the sign flip cancels the e^{i pi/2})
        return (a + 4 * self.N) % (8 * self.N)  # extra -1

    def beta_exp(self, g) -> int:
        """beta on the chosen boundary subgroups.

        K1 = <rs>, K2 = <s> : alpha trivial there, beta == 1.
        K3 = <r>           : beta = beta_N above.
        Defined for the elements that actually occur on each
        boundary; raises for anything off the rotation axis with j=1
        other than the order-2 reflections used in the encoding.
        """
        a, j = g
        if j == 0:
            return self.beta_rot_exp(a)
        # reflections: only used on K1=<rs>, K2=<s> where beta == 1
        return 0


# ----------------------------------------------------------------
# Verification helpers -- the "verify against canonical" layer.
# ----------------------------------------------------------------
def check_cocycle_condition(coc: DihedralCocycle, which: str = "alpha") -> bool:
    """alpha(g,h) alpha(gh,k) = alpha(g,hk) alpha(h,k)  (Eq. A.6).

    Checked exactly in integer-exponent arithmetic over all triples.
    `which` selects 'alpha' (normalized) or 'raw' (alpha'_N).
    """
    G = coc.G
    f = coc.alpha_exp if which == "alpha" else coc.alpha_raw_exp
    M = 8 * coc.N
    for g in G.elements():
        for h in G.elements():
            gh = G.mul(g, h)
            for k in G.elements():
                hk = G.mul(h, k)
                lhs = (f(g, h) + f(gh, k)) % M
                rhs = (f(g, hk) + f(h, k)) % M
                if lhs != rhs:
                    return False
    return True


def check_normalization(coc: DihedralCocycle) -> bool:
    """Beigi-Shor-Whalen normalization of alpha_N (paper Eq. A.18):

        alpha(id,g) = alpha(g,id) = 1,
        alpha(g, g^-1) = 1,
        alpha(h^-1, g^-1) = alpha(g,h)^-1.
    """
    G, ph = coc.G, coc.ph
    idg = G.identity()
    for g in G.elements():
        if not ph.is_one(coc.alpha_exp(idg, g)):
            return False
        if not ph.is_one(coc.alpha_exp(g, idg)):
            return False
        if not ph.is_one(coc.alpha_exp(g, G.inverse(g))):
            return False
    for g in G.elements():
        for h in G.elements():
            lhs = coc.alpha_exp(G.inverse(h), G.inverse(g))
            rhs = ph.inv(coc.alpha_exp(g, h))
            if lhs != rhs:
                return False
    return True


def check_beta_trivializes(coc: DihedralCocycle, subgroup="r") -> bool:
    """Verify alpha_N|_K = delta beta_N on K (paper Eq. A.29 / II.19).

        alpha(g,h) = beta(g) beta(h) / beta(gh)  for all g,h in K.
    """
    G, ph = coc.G, coc.ph
    if subgroup == "r":
        K = [G.rot(a) for a in range(G.n_rot)]
    elif subgroup == "s":
        K = [G.identity(), G.s]
    elif subgroup == "rs":
        K = [G.identity(), G.rs]
    else:
        raise ValueError(subgroup)
    M = 8 * coc.N
    for g in K:
        for h in K:
            gh = G.mul(g, h)
            db = (coc.beta_exp(g) + coc.beta_exp(h) - coc.beta_exp(gh)) % M
            if db != coc.alpha_exp(g, h):
                return False
    return True


def check_raw_formulas(coc: DihedralCocycle) -> dict:
    """Independent checks against the paper's *explicit* closed forms
    for the un-normalized cocycle (App. A.3), not just self-consistency:

      * alpha'_N(r^{2N}, r^{2N}) = -1                       (Eq. A.24)
      * alpha'_N|<r> = delta beta'_N, beta'_N(r^a)=e^{i pi a/4N} (A.25)
      * beta_N(r) = e^{i pi/(4N)},  beta_N(r^{2N}) = +1        (A.29)
    """
    G, ph = coc.G, coc.ph
    M = 8 * coc.N
    out = {}
    # A.24
    out["alpha'(r^2N,r^2N) == -1"] = (
        coc.alpha_raw_exp(G.rot(2 * coc.N), G.rot(2 * coc.N)) == 4 * coc.N
    )
    # A.25: raw cocycle trivialized by beta' on <r>
    ok = True
    for a in range(G.n_rot):
        for b in range(G.n_rot):
            ga, gb = G.rot(a), G.rot(b)
            gab = G.mul(ga, gb)
            db = (coc.beta_raw_rot_exp(a) + coc.beta_raw_rot_exp(b)
                  - coc.beta_raw_rot_exp(gab[0])) % M
            if db != coc.alpha_raw_exp(ga, gb):
                ok = False
                break
        if not ok:
            break
    out["alpha'|<r> == delta beta'  (A.25)"] = ok
    # A.29 specific values
    out["beta_N(r) == e^{i pi/(4N)}"] = (
        abs(ph.to_complex(coc.beta_rot_exp(1))
            - cmath.exp(1j * math.pi / (4 * coc.N))) < 1e-12
    )
    out["beta_N(r^{2N}) == +1"] = ph.is_one(coc.beta_rot_exp(2 * coc.N))
    return out


def check_class_order_two(coc: DihedralCocycle) -> bool:
    """H^2(D_{4N},U(1)) = Z_2: the class of alpha_N has order exactly 2.

    We use the cohomologous *raw* representative alpha'_N (=alpha_N up
    to the coboundary delta kappa), which is genuinely {+-1}-valued and
    hence 2-torsion as a cocycle: alpha'_N^2 is identically trivial, so
    2*[alpha_N] = 0.  Combined with [alpha_N] != 0 (alpha_N not a
    coboundary), the class order is exactly 2.

    (The normalized alpha_N itself takes values in {+-1, +-i} because of
    the kappa renormalization, so alpha_N^2 is only a coboundary, not the
    identity cocycle -- which is why the raw representative is the clean
    witness for 2-torsion.)
    """
    M = 8 * coc.N
    squared_trivial = all(
        (2 * coc.alpha_raw_exp(g, h)) % M == 0
        for g in coc.G.elements()
        for h in coc.G.elements()
    )
    return squared_trivial and not is_coboundary(coc)


def is_coboundary(coc: DihedralCocycle) -> bool:
    """Is alpha_N a coboundary (trivial class)?  Brute-force search
    for a global 1-cochain kappa: G -> U(1) with alpha = delta kappa.

    The U(1) values needed here are 8N-th roots of unity (alpha is
    2-torsion so a trivializing cochain can be taken valued in
    8N-th roots), so this is a finite, exact search via linear
    algebra over Z/(8N): solve  kappa(g)+kappa(h)-kappa(gh) = alpha
    for the unknown integer exponents kappa(.).  Returns True iff a
    solution exists.  For the non-trivial class it must return False.
    """
    G = coc.G
    M = 8 * coc.N
    els = G.elements()
    idx = {g: i for i, g in enumerate(els)}
    n = len(els)
    # Build the linear system over Z/M:  for each (g,h),
    #   x[g] + x[h] - x[gh] == alpha(g,h)   (mod M)
    # Gauge-fix x[id] = 0.  Solve by integer Gaussian elimination
    # mod M (M = 8N is not prime, so use a careful elimination that
    # only divides by units; if it gets stuck we fall back to a
    # consistency check).  We instead use a simpler robust route:
    # propagate constraints from generators.
    #
    # x is determined on generators r, s by the relations; check all.
    # Try every assignment of x[r], x[s] in Z/M (M*M cases) — exact,
    # finite, and small for the N we use.
    r, s = G.r, G.s
    for xr in range(M):
        for xs in range(M):
            x = [None] * n
            x[idx[G.identity()]] = 0
            # generate the whole group from words; assign via the
            # cocycle relation x[gh] = x[g] + x[h] - alpha(g,h)
            x[idx[r]] = xr
            x[idx[s]] = xs
            ok = True
            changed = True
            while changed and ok:
                changed = False
                for g in els:
                    ig = idx[g]
                    if x[ig] is None:
                        continue
                    for gen, xg in ((r, xr), (s, xs)):
                        gh = G.mul(g, gen)
                        igh = idx[gh]
                        val = (x[ig] + xg - coc.alpha_exp(g, gen)) % M
                        if x[igh] is None:
                            x[igh] = val
                            changed = True
                        elif x[igh] != val:
                            ok = False
                            break
                    if not ok:
                        break
            if not ok or any(v is None for v in x):
                continue
            # full consistency check over all pairs
            good = True
            for g in els:
                for h in els:
                    gh = G.mul(g, h)
                    if (x[idx[g]] + x[idx[h]] - x[idx[gh]]) % M != coc.alpha_exp(g, h):
                        good = False
                        break
                if not good:
                    break
            if good:
                return True
    return False


# ----------------------------------------------------------------
# The transversal phase U_{alpha,beta} (paper Theorem 1, Eq. II.22)
# and the canonical gate-table check (Corollary 1 / Table I).
# ----------------------------------------------------------------
def U_alpha_beta_exp(coc: DihedralCocycle, g1, g2,
                     b1=None, b2=None, b3=None) -> int:
    """U_{alpha,beta}(g1,g2) phase exponent (paper Eq. II.22):

        U(g1,g2) = alpha(g1,g2) * b3(g1 g2) / ( b1(g1) b2(g2) ).

    b1, b2, b3 are 1-cochains on the three boundaries; if omitted
    they default to the paper's choice for D_{4N}:
        b1 on K1=<rs> : == 1
        b2 on K2=<s>  : == 1
        b3 on K3=<r>  : == beta_N.
    """
    G = coc.G
    if b1 is None:
        b1 = lambda g: 0           # noqa: E731  (== 1)
    if b2 is None:
        b2 = lambda g: 0           # noqa: E731
    if b3 is None:
        b3 = coc.beta_exp
    g1g2 = G.mul(g1, g2)
    M = 8 * coc.N
    return (coc.alpha_exp(g1, g2) + b3(g1g2) - b1(g1) - b2(g2)) % M


def reproduce_gate_table(N: int):
    """Reproduce the paper's logical gate (Corollary 1 / Table I).

    With the encoding |1> <-> (g1,g2) = (rs, s) (paper Eq. III.36),
    U_{alpha,beta}(rs,s) = beta_N(r) = exp(i pi/(4N)), i.e. the gate
        T^{1/N} = P(pi/(4N)) = diag(1, exp(i pi/(4N))).
    Returns a dict with the realized phase and its identification.
    """
    coc = DihedralCocycle(N)
    G = coc.G
    e0 = U_alpha_beta_exp(coc, G.identity(), G.identity())  # |0> state
    e1 = U_alpha_beta_exp(coc, G.rs, G.s)                    # |1> state
    ph = coc.ph
    # paper: 8N = 2^n  =>  Clifford-hierarchy level n
    n = None
    if (8 * N) & (8 * N - 1) == 0:  # power of two
        n = (8 * N).bit_length() - 1
    return {
        "N": N,
        "group": f"D_{4*N} (order {8*N})",
        "phase0_exp": e0,
        "phase1_exp": e1,
        "phase1": ph.to_complex(e1),
        "expected_phase1": cmath.exp(1j * math.pi / (4 * N)),
        "gate": f"T^(1/{N}) = P(pi/{4*N}) = diag(1, exp(i*pi/{4*N}))",
        "clifford_level_n": n,
        "matches": abs(ph.to_complex(e1) - cmath.exp(1j * math.pi / (4 * N))) < 1e-12
        and ph.is_one(e0),
    }


if __name__ == "__main__":
    print("Dihedral 2-cocycle verification  (Warman-Schafer-Nameki 2512.13777)\n")
    for N in (1, 2, 4, 8, 16):
        coc = DihedralCocycle(N)
        tab = reproduce_gate_table(N)
        print(f"D_{4*N}  (order {8*N}, N={N}):")
        print(f"  cocycle condition (alpha'_N) : {check_cocycle_condition(coc,'raw')}")
        print(f"  cocycle condition (alpha_N)  : {check_cocycle_condition(coc,'alpha')}")
        print(f"  normalization (Eq. A.18)     : {check_normalization(coc)}")
        print(f"  beta trivializes on <r>      : {check_beta_trivializes(coc,'r')}")
        raw = check_raw_formulas(coc)
        print(f"  raw closed-forms A.24/A.25/A.29: {all(raw.values())}  {raw}")
        print(f"  class order exactly 2 (Z_2)  : {check_class_order_two(coc)}")
        print(f"  gate U(rs,s) = {tab['phase1']:.6f}  -> {tab['gate']}")
        print(f"  matches paper Table I        : {tab['matches']}   "
              f"(Clifford level n={tab['clifford_level_n']})\n")
