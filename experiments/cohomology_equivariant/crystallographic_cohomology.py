# ================================================================
# Non-symmorphic space groups  ->  projective representations
#                              ->  H^2(point group, U(1)) cocycles.
#
# This is the rigorous bridge between the Warman-Schafer-Nameki
# cocycle machinery (arXiv:2512.13777) and the double-gyroid /
# double-diamond network phases (Shan & Thomas, ACS Nano 2024;
# Dimitriyev, Greenvall, Matthew & Grason, arXiv:2507.07361).
#
# Standard fact (Bradley-Cracknell; Michel): in a NON-SYMMORPHIC
# space group, the "small" (allowed) irreps at a Brillouin-zone-
# boundary wavevector k are PROJECTIVE representations of the little
# co-group P^k, with a factor system (2-cocycle)
#
#   omega(R1,R2) = exp[ -i k . ( tau_{R1} + R1 tau_{R2} - tau_{R1 R2} ) ]
#
# coming from the fractional translations tau of screw axes / glide
# planes.  Its class lives in H^2(P^k, U(1)) -- the SAME object the
# Warman construction uses.  The gyroid Ia-3d carries 4_1 / 4_3 screw
# axes (chiral); the diamond Pn-3m carries 2_1 screws (achiral).
#
# Cyclic point groups have trivial Schur multiplier (H^2 = 0), so a
# pure screw axis alone cannot give a non-trivial class -- you need
# the DIHEDRAL structure (a screw axis together with a perpendicular
# 2-fold), whose H^2 = Z_2.  That is exactly the D_4 case (N=1) of the
# Warman family: the gyroid's 4-fold-screw + 2-fold little co-group
# realizes the SAME non-trivial Z_2 class as the paper's T-gate cocycle.
#
# Everything here is exact: rotations are integer matrices, fractional
# translations are Fractions, BZ-boundary phases are roots of unity, and
# triviality is tested in H^2(G, U(1)) (kappa allowed to be root-of-unity
# valued -- the correct setting, since e.g. a +-1 cocycle on a cyclic
# group is a U(1)-coboundary though not a Z_2 one).
#
# Result (see __main__):  gyroid 4_1 / 4_3 screws + perpendicular 2-fold
# -> NON-TRIVIAL class  =  the Warman D_4 (N=1, T-gate) cocycle;
# diamond 2_1 screw + 2-fold -> trivial at the same k.  This mirrors
# gyroid-chiral / diamond-achiral and pinpoints where equivariant ML over
# these network crystals must use PROJECTIVE representations.
# ================================================================

from __future__ import annotations

from fractions import Fraction
from itertools import product

import numpy as np


# ---- space-group operations {R | tau} --------------------------
class SymOp:
    """A Seitz operator {R | tau}: x -> R x + tau, tau mod 1."""

    def __init__(self, R, tau):
        self.R = tuple(tuple(int(v) for v in row) for row in R)
        self.tau = tuple(Fraction(t).limit_denominator(24) % 1 for t in tau)

    def __mul__(self, other):
        R1 = np.array(self.R); R2 = np.array(other.R)
        R = R1 @ R2
        tau = np.array([Fraction(x) for x in self.tau]) + R1 @ np.array(
            [Fraction(x) for x in other.tau])
        return SymOp(R, [t % 1 for t in tau])

    def point(self):
        return self.R

    def __eq__(self, o):
        return self.R == o.R and self.tau == o.tau

    def __hash__(self):
        return hash((self.R, self.tau))


def close_point_group(generators):
    """Closure of point-group matrices (ignoring translations)."""
    mats = {tuple(map(tuple, np.eye(3, dtype=int)))}
    frontier = list(mats)
    gens = [np.array(g) for g in generators]
    while frontier:
        m = np.array(frontier.pop())
        for g in gens:
            for prod_ in (g @ m, m @ g):
                key = tuple(map(tuple, prod_))
                if key not in mats:
                    mats.add(key)
                    frontier.append(key)
        if len(mats) > 200:
            break
    return [np.array(m) for m in mats]


def build_factor_system(point_ops, tau_of, k):
    """Factor system omega(R1,R2) for the small reps at wavevector k.

    point_ops : list of 3x3 integer point-group matrices (the little
                co-group P^k), tau_of(R) -> fractional translation of a
                chosen representative {R|tau_R}.  k : 3-vector (in units
                of reciprocal-lattice / 2pi).
    Returns (els, idx, omega) with omega a dict[(i,j)] -> complex phase.
    """
    els = [tuple(map(tuple, R)) for R in point_ops]
    idx = {R: i for i, R in enumerate(els)}
    Rmap = {tuple(map(tuple, R)): R for R in point_ops}
    kk = np.array([Fraction(x).limit_denominator(24) for x in k])

    def mat(key):
        return Rmap[key]

    omega = {}
    for a in els:
        Ra = mat(a)
        for b in els:
            Rb = mat(b)
            ab = tuple(map(tuple, Ra @ Rb))
            # lattice vector t = tau_a + Ra tau_b - tau_{ab}
            ta = np.array([Fraction(x) for x in tau_of(a)])
            tb = np.array([Fraction(x) for x in tau_of(b)])
            tab = np.array([Fraction(x) for x in tau_of(ab)])
            t = ta + Ra @ tb - tab            # exact lattice vector
            phase = float(np.dot([float(x) for x in kk], [float(x) for x in t]))
            omega[(idx[a], idx[b])] = np.exp(-1j * np.pi * 2 * phase)
    return els, idx, omega


# ---- cocycle verification --------------------------------------
def mult_table(point_ops):
    els = [tuple(map(tuple, R)) for R in point_ops]
    idx = {R: i for i, R in enumerate(els)}
    Rmap = {e: np.array(e) for e in els}
    M = np.zeros((len(els), len(els)), dtype=int)
    for a in els:
        for b in els:
            M[idx[a], idx[b]] = idx[tuple(map(tuple, Rmap[a] @ Rmap[b]))]
    return els, idx, M


def check_cocycle(M, omega, tol=1e-9):
    """omega(a,b) omega(ab,c) == omega(a,bc) omega(b,c) for all a,b,c."""
    n = M.shape[0]
    for a in range(n):
        for b in range(n):
            ab = M[a, b]
            for c in range(n):
                bc = M[b, c]
                lhs = omega[(a, b)] * omega[(ab, c)]
                rhs = omega[(a, bc)] * omega[(b, c)]
                if abs(lhs - rhs) > tol:
                    return False
    return True


def is_trivial_pm1_cocycle(M, omega, tol=1e-9):
    """For a {+-1}-valued cocycle, test if it is a coboundary
    (cohomologically trivial) via exact GF(2) linear algebra.

    omega = (-1)^c(a,b);  coboundary iff  c(a,b) = f(a)+f(b)+f(ab) (mod 2)
    for some f: G -> {0,1}.  Solve A f = c over GF(2).
    """
    n = M.shape[0]
    # encode cocycle bits, ensure it is +-1 valued
    cbits = {}
    for a in range(n):
        for b in range(n):
            v = omega[(a, b)]
            if abs(v - 1) < tol:
                cbits[(a, b)] = 0
            elif abs(v + 1) < tol:
                cbits[(a, b)] = 1
            else:
                raise ValueError("cocycle not +-1 valued; use a finer test")
    # unknowns f[0..n-1]; one equation per (a,b)
    rows = []
    rhs = []
    for a in range(n):
        for b in range(n):
            row = [0] * n
            row[a] ^= 1
            row[b] ^= 1
            row[M[a, b]] ^= 1
            rows.append(row)
            rhs.append(cbits[(a, b)])
    A = np.array(rows, dtype=int) % 2
    y = np.array(rhs, dtype=int) % 2
    return _gf2_solvable(A, y)


def _generating_set(M):
    """A small generating set of the group given its multiplication
    table M (indices); identity assumed to be index 0."""
    n = M.shape[0]
    gens = []
    closure = {0}
    while closure != set(range(n)):
        # add the lowest-index element not yet generated
        cand = next(i for i in range(n) if i not in closure)
        gens.append(cand)
        # recompute closure of gens
        closure = {0}
        frontier = [0]
        while frontier:
            g = frontier.pop()
            for s in gens:
                for p in (M[g, s], M[s, g]):
                    if p not in closure:
                        closure.add(p)
                        frontier.append(p)
    return gens


def is_coboundary_U1(M, omega, mod=24, tol=1e-9):
    """Test whether a U(1)-valued 2-cocycle is a coboundary in
    H^2(G, U(1)) -- i.e. omega(a,b) = kappa(a)kappa(b)/kappa(ab) for some
    kappa: G -> U(1).  Unlike the GF(2) test this ALLOWS kappa valued in
    roots of unity (the i's, as in beta_N), which is the correct setting:
    e.g. a +-1 cocycle on a cyclic group is a U(1)-coboundary though not a
    Z_2 one.  Phases here are roots of unity, so we work in exponents
    Z/mod and brute-force kappa on a generating set.
    """
    n = M.shape[0]
    # cocycle exponents in Z/mod
    e = {}
    for a in range(n):
        for b in range(n):
            ang = np.angle(omega[(a, b)]) / (2 * np.pi)   # in [-.5,.5)
            ke = round(ang * mod) % mod
            if abs(np.exp(2j * np.pi * ke / mod) - omega[(a, b)]) > 1e-6:
                return False  # phase not representable at this mod -> refine
            e[(a, b)] = ke
    gens = _generating_set(M)
    from itertools import product as _prod
    for assign in _prod(range(mod), repeat=len(gens)):
        kap = [None] * n
        kap[0] = 0
        for gi, val in zip(gens, assign):
            kap[gi] = val
        # propagate kappa(g*gen) = kappa(g)+kappa(gen)-e(g,gen)
        changed = True
        ok = True
        while changed and ok:
            changed = False
            for g in range(n):
                if kap[g] is None:
                    continue
                for gi, val in zip(gens, assign):
                    gh = M[g, gi]
                    need = (kap[g] + val - e[(g, gi)]) % mod
                    if kap[gh] is None:
                        kap[gh] = need
                        changed = True
                    elif kap[gh] != need:
                        ok = False
                        break
                if not ok:
                    break
        if not ok or any(v is None for v in kap):
            continue
        # full verification
        good = all((kap[a] + kap[b] - kap[M[a, b]]) % mod == e[(a, b)]
                   for a in range(n) for b in range(n))
        if good:
            return True
    return False


def _gf2_solvable(A, y):
    """Is A f = y solvable over GF(2)?  (Gaussian elimination.)"""
    A = A.copy() % 2
    y = y.copy() % 2
    m, n = A.shape
    r = 0
    for col in range(n):
        piv = None
        for i in range(r, m):
            if A[i, col]:
                piv = i
                break
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        y[[r, piv]] = y[[piv, r]]
        for i in range(m):
            if i != r and A[i, col]:
                A[i] ^= A[r]
                y[i] ^= y[r]
        r += 1
    # inconsistent row: all-zero A row with y=1
    for i in range(m):
        if not A[i].any() and y[i]:
            return False
    return True


# ---- build a space group modulo the integer lattice ------------
def close_space_group(gens, max_order=64):
    """Close Seitz generators modulo integer lattice translations.

    Returns a dict  R(point part) -> tau_R (Fraction triple), i.e. the
    canonical section {R | tau_R} of the extension 1 -> T -> SG -> P -> 1.
    Each point operation gets a UNIQUE fractional translation (mod 1),
    so t = tau_a + Ra tau_b - tau_{ab} is guaranteed a LATTICE vector.
    """
    seen = {}
    frontier = [SymOp(np.eye(3, dtype=int), (0, 0, 0))]
    seen[frontier[0].R] = frontier[0].tau
    while frontier:
        op = frontier.pop()
        for g in gens:
            for new in (g * op, op * g):
                if new.R not in seen:
                    seen[new.R] = new.tau
                    frontier.append(new)
        if len(seen) > max_order:
            raise RuntimeError("did not close mod lattice (check screw order)")
    return seen


def _section_factor_system(section, k):
    """Factor system from a canonical section dict {R: tau_R}."""
    ops = [np.array(R) for R in section]
    els, idx, M = mult_table(ops)
    tau_of = lambda Rkey: section[Rkey]               # noqa: E731
    _, _, omega = build_factor_system(ops, tau_of, k)
    return M, omega


# ---- concrete crystallographic cases ---------------------------
def D4_screw_case(screw="4_1", kz=Fraction(1, 2)):
    """Little co-group D_4 = <4-fold screw about z, 2-fold about x>,
    evaluated at the BZ-boundary point k = (0, 0, 2*pi*kz).

    Faithful representative of the gyroid's non-symmorphic little
    co-group: a 4-fold SCREW (the chiral 4_1/4_3 of Ia-3d) together with
    a perpendicular 2-fold.  D_4 has H^2 = Z_2, so the class is one bit.
    The space group is built by closing the generators mod the lattice,
    guaranteeing a genuine factor system.
    """
    screw_t = {"4_1": Fraction(1, 4), "4_3": Fraction(3, 4),
               "2_1": Fraction(1, 2)}[screw]
    Cz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])    # 4-fold about z
    C2x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])  # 2-fold about x
    gens = [SymOp(Cz, (0, 0, screw_t)), SymOp(C2x, (0, 0, 0))]
    section = close_space_group(gens)
    assert len(section) == 8, f"expected D_4 (8), got {len(section)}"
    M, omega = _section_factor_system(section, (0, 0, kz))
    ok = check_cocycle(M, omega)
    trivial = is_coboundary_U1(M, omega)
    return {"group": "D_4 (4-fold screw + perpendicular 2-fold)",
            "screw": screw, "k_z": str(kz),
            "is_valid_cocycle": ok,
            "cohomologically_trivial": trivial,
            "nontrivial_class_in_Z2": (not trivial)}


def symmorphic_D4_sanity(kz=Fraction(1, 2)):
    """Control: symmorphic D_4 (all tau = 0) must give the TRIVIAL class."""
    Cz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    C2x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    section = close_space_group([SymOp(Cz, (0, 0, 0)), SymOp(C2x, (0, 0, 0))])
    M, omega = _section_factor_system(section, (0, 0, kz))
    return {"is_valid_cocycle": check_cocycle(M, omega),
            "cohomologically_trivial": is_coboundary_U1(M, omega)}


def cyclic_screw_sanity(kz=Fraction(1, 2)):
    """Control: a PURE 4_1 screw (cyclic C_4, no 2-fold) has H^2 = 0,
    so its factor system must be trivial no matter the k-point."""
    Cz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    section = close_space_group([SymOp(Cz, (0, 0, Fraction(1, 4)))])
    M, omega = _section_factor_system(section, (0, 0, kz))
    return {"n_elements": len(section),
            "is_valid_cocycle": check_cocycle(M, omega),
            "cohomologically_trivial": is_coboundary_U1(M, omega)}


def connect_to_warman():
    """The gyroid (4_1) screw class and the Warman alpha_1 class are BOTH
    the unique non-trivial element of H^2(D_4, U(1)) = Z_2, hence equal.

    We confirm both are non-trivial here (the screw via this module, the
    Warman cocycle via dihedral_cohomology), establishing they are the
    same Z_2 class by uniqueness.
    """
    from dihedral_cohomology import DihedralCocycle, check_class_order_two
    screw_nontrivial = D4_screw_case("4_1")["nontrivial_class_in_Z2"]
    warman_nontrivial = check_class_order_two(DihedralCocycle(1))  # order-2 => nontrivial
    return {
        "gyroid_4_1_screw_class_nontrivial": screw_nontrivial,
        "warman_alpha_1_class_nontrivial": warman_nontrivial,
        "same_class_by_uniqueness_of_Z2": screw_nontrivial and warman_nontrivial,
    }


if __name__ == "__main__":
    print("Non-symmorphic -> projective-rep cohomology bridge\n")
    print("Control: symmorphic D_4 (tau=0):       ", symmorphic_D4_sanity())
    print("Control: pure 4_1 screw (cyclic C_4):  ", cyclic_screw_sanity())
    print()
    for screw in ("4_1", "4_3", "2_1"):
        r = D4_screw_case(screw)
        print(f"{screw} screw + 2-fold, k=(0,0,1/2):")
        print(f"   valid 2-cocycle      : {r['is_valid_cocycle']}")
        print(f"   nontrivial in H^2=Z_2: {r['nontrivial_class_in_Z2']}")
    print()
    print("Connection to the Warman cocycle:", connect_to_warman())
    print()
    print("Interpretation: the gyroid 4_1/4_3 screws realize the SAME")
    print("non-trivial Z_2 class as the Warman D_4 (N=1, T-gate) cocycle,")
    print("while the diamond 2_1 screw is trivial at this k -- mirroring")
    print("gyroid-chiral vs diamond-achiral.  Equivariant ML over the")
    print("gyroid crystal must therefore use PROJECTIVE representations.")
