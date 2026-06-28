# ================================================================
# Projective-equivariant ML substrate from H^2(G, U(1)).
#
# This is the ML transfer of the Warman-Schafer-Nameki construction
# (arXiv:2512.13777). The bridge is a textbook fact of geometric
# deep learning that the QEC paper makes operational:
#
#     H^2(G, U(1)) classifies the PROJECTIVE representations of G.
#
# A standard G-equivariant layer (group convolution) is an
# intertwiner of an *ordinary* (linear) representation. If the data
# instead carries a *projective* symmetry with a non-trivial cocycle
# alpha, then the linear-equivariant hypothesis space provably does
# NOT contain the target intertwiner -- this is the cohomological
# obstruction.  The fix is exactly the paper's ingredients:
#
#   alpha_N   -> the cocycle defining the (twisted) data symmetry,
#   rho_alpha -> the alpha-twisted regular representation,
#   R^alpha_g -> the alpha-twisted group-convolution operators
#                (the equivariant projection / "U_{alpha,beta}"),
#   beta_N    -> the 1-cochain whose diagonal gauge D_beta trivializes
#                rho_alpha on the cyclic subgroup <r>, recovering an
#                ordinary-equivariant model there.
#
# Operators are |G| x |G| complex matrices acting on C[G] (the same
# per-edge Hilbert space H = C[G] used in the paper, Sec. II B).
# ================================================================

from __future__ import annotations

import numpy as np

from dihedral_cohomology import Dihedral, DihedralCocycle


class CocycleRepresentation:
    """Ordinary and alpha-twisted regular reps of D_{4N} on C[G],
    plus the matching (twisted) group-convolution operators.

    Set ``cocycle=None`` for the trivial class (alpha == 1): then the
    twisted objects collapse onto the ordinary ones, which is the
    control used to prove the effect is the cohomology class and not
    model capacity.
    """

    def __init__(self, N: int, cocycle: bool = True):
        self.N = N
        self.G = Dihedral(N)
        self.dim = self.G.order
        self.els = self.G.elements()
        self.idx = {g: i for i, g in enumerate(self.els)}
        self.coc = DihedralCocycle(N) if cocycle else None

    # ---- phase lookups -----------------------------------------
    def _alpha(self, g, h) -> complex:
        if self.coc is None:
            return 1.0 + 0.0j
        return self.coc.alpha(g, h)

    # ---- left regular representations ---------------------------
    def L(self, g) -> np.ndarray:
        """Ordinary left-regular rep:  L(g)|h> = |g h>  (paper II.2)."""
        M = np.zeros((self.dim, self.dim), dtype=complex)
        for h in self.els:
            M[self.idx[self.G.mul(g, h)], self.idx[h]] = 1.0
        return M

    def L_alpha(self, g) -> np.ndarray:
        """alpha-twisted left-regular rep:  rho_alpha(g)|h> = alpha(g,h)|g h>.

        Satisfies rho_alpha(g) rho_alpha(h) = alpha(g,h) rho_alpha(gh):
        a genuine projective representation with cocycle alpha.
        """
        M = np.zeros((self.dim, self.dim), dtype=complex)
        for h in self.els:
            M[self.idx[self.G.mul(g, h)], self.idx[h]] = self._alpha(g, h)
        return M

    # ---- group-convolution operators (the equivariant layers) --
    def R(self, g) -> np.ndarray:
        """Ordinary right-regular operator:  R(g)|h> = |h g^-1>.

        Commutes with every L(k): the ordinary group-convolution
        algebra (= commutant of the linear regular rep).
        """
        M = np.zeros((self.dim, self.dim), dtype=complex)
        ginv = self.G.inverse(g)
        for h in self.els:
            M[self.idx[self.G.mul(h, ginv)], self.idx[h]] = 1.0
        return M

    def R_alpha(self, g) -> np.ndarray:
        """alpha-twisted right-regular operator:

            R^alpha(g)|h> = alpha(h, g^-1) |h g^-1>.

        Derived so that [L_alpha(k), R^alpha(g)] = 0 for all k (uses
        the 2-cocycle identity).  These span the commutant of the
        alpha-twisted regular rep -- the alpha-twisted convolution
        algebra, i.e. the cocycle-aware equivariant layer.
        """
        M = np.zeros((self.dim, self.dim), dtype=complex)
        ginv = self.G.inverse(g)
        for h in self.els:
            M[self.idx[self.G.mul(h, ginv)], self.idx[h]] = self._alpha(h, ginv)
        return M

    # ---- bases of the two equivariant hypothesis spaces --------
    def conv_basis(self, twisted: bool) -> np.ndarray:
        """Stack the |G| convolution operators into a (|G|, d, d) array.

        twisted=False -> ordinary conv {R(g)}  (linear-equivariant).
        twisted=True  -> alpha-twisted conv {R^alpha(g)} (cocycle-aware).
        Both have |G| complex parameters: the ONLY difference is the
        cohomology class twisting the operators.
        """
        op = self.R_alpha if twisted else self.R
        return np.stack([op(g) for g in self.els], axis=0)

    # ---- beta-gauge that trivializes rho_alpha on <r> ----------
    def beta_gauge(self) -> np.ndarray:
        """Diagonal gauge D_beta = diag(beta(h)) (paper's beta_N).

        Key identity, SUBGROUP-LOCAL on the C[K] block, K = <r>:
            D_beta @ L_alpha(k) @ D_beta^{-1} = beta(k) * L(k)   for k in <r>,
        so conjugation by D_beta turns the twisted regular rep into the
        ordinary one (up to the harmless scalar beta(k)) on the cyclic
        subgroup.  It is necessarily only subgroup-local: alpha_N is a
        non-trivial class, so NO global trivialization exists -- that is
        the obstruction itself.  On C[K] the twisted-equivariant problem
        becomes an ordinary-equivariant one after gauging -- the ML form
        of "find the trivialization on a subgroup, get the equivariant
        projection" (paper Thm 1, Eq. II.19).  Verified in tests and in
        run_obstruction_experiment.py (Experiment C).
        """
        if self.coc is None:
            return np.eye(self.dim, dtype=complex)
        d = np.array([self.coc.ph.to_complex(self.coc.beta_exp(h)) for h in self.els])
        return np.diag(d)

    def rot_subspace_indices(self):
        """Indices of the cyclic subgroup <r> = Z_{4N} within C[G]."""
        return [self.idx[self.G.rot(a)] for a in range(self.G.n_rot)]


# ----------------------------------------------------------------
# Generic commutant + obstruction geometry (numerical, exact-ish).
# ----------------------------------------------------------------
def commutant_basis(generators, tol: float = 1e-9) -> np.ndarray:
    """Orthonormal basis (as vectorized matrices) of {T : [T, M] = 0
    for all M in `generators`}.  Solves the stacked linear system via
    the right singular vectors with ~zero singular value.
    """
    d = generators[0].shape[0]
    rows = []
    eye = np.eye(d)
    for M in generators:
        # row-major (numpy C-order) vec convention, matching conv_basis:
        #   vec([T,M]) = (I (x) M^T - M (x) I) vec(T) = 0
        rows.append(np.kron(eye, M.T) - np.kron(M, eye))
    A = np.vstack(rows)
    # nullspace of A
    _, sv, Vh = np.linalg.svd(A)
    # singular values count: A is (k*d^2, d^2); pad sv to d^2
    full = np.zeros(d * d)
    full[: len(sv)] = sv
    null_mask = full < tol
    basis = Vh.conj().T[:, null_mask]  # (d^2, dim_null)
    return basis  # columns are vectorized matrices


def subspace_overlap(basisA: np.ndarray, basisB: np.ndarray):
    """Principal-angle diagnostics between two subspaces given by
    orthonormal column bases (vectorized matrices).

    Returns (dim_intersection, n_shared_directions, mean_cos2),
    where mean_cos2 in [0,1] is the average squared cosine of the
    principal angles: 1 == identical subspaces, 0 == orthogonal.
    """
    # orthonormalize defensively
    QA, _ = np.linalg.qr(basisA)
    QB, _ = np.linalg.qr(basisB)
    M = QA.conj().T @ QB
    sv = np.linalg.svd(M, compute_uv=False)
    cos2 = np.clip(sv ** 2, 0, 1)
    dim_inter = int(np.sum(cos2 > 1 - 1e-6))
    return dim_inter, len(sv), float(np.mean(cos2))


def project_onto(basis: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Orthogonal projection of `vec` onto span(columns of basis)."""
    Q, _ = np.linalg.qr(basis)
    return Q @ (Q.conj().T @ vec)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    for N in (1, 2):
        print(f"\n=== D_{4*N} (order {8*N}) ===")
        rep = CocycleRepresentation(N, cocycle=True)
        G = rep.G

        # 1) rho_alpha is a genuine projective rep with cocycle alpha
        max_err = 0.0
        for g in G.elements():
            for h in G.elements():
                lhs = rep.L_alpha(g) @ rep.L_alpha(h)
                rhs = rep.coc.alpha(g, h) * rep.L_alpha(G.mul(g, h))
                max_err = max(max_err, np.max(np.abs(lhs - rhs)))
        print(f"projective rep  rho_a(g)rho_a(h)=alpha(g,h)rho_a(gh):  max err = {max_err:.2e}")

        # 2) twisted conv operators commute with the twisted rep
        max_comm = 0.0
        for g in G.elements():
            Rg = rep.R_alpha(g)
            for k in (G.r, G.s, G.rs):
                Lk = rep.L_alpha(k)
                max_comm = max(max_comm, np.max(np.abs(Lk @ Rg - Rg @ Lk)))
        print(f"[L_alpha(k), R_alpha(g)] = 0  (intertwiner):          max err = {max_comm:.2e}")

        # 3) twisted conv basis == numerical commutant of rho_alpha
        Hlin = rep.conv_basis(twisted=False).reshape(rep.dim, -1).T
        Halpha = rep.conv_basis(twisted=True).reshape(rep.dim, -1).T
        comm_alpha = commutant_basis([rep.L_alpha(G.r), rep.L_alpha(G.s)])
        di, _, c2 = subspace_overlap(Halpha, comm_alpha)
        print(f"twisted conv spans commutant(rho_alpha):  mean cos^2 = {c2:.6f}")

        # 4) THE OBSTRUCTION: how different are the two hypothesis spaces
        dim_inter, ntot, mean_cos2 = subspace_overlap(Hlin, Halpha)
        print(f"linear vs twisted conv:  dim={ntot}, shared dims={dim_inter}, "
              f"mean cos^2(angles)={mean_cos2:.4f}")
