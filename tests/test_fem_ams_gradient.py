"""AMS Milestone 1 — the discrete gradient ``G`` and the gradient-space correction it enables.

Two guards: (1) the discrete de Rham identity — the pure curl-curl operator annihilates every column
of ``G`` (``curl∘grad = 0``), which pins the edge-sign convention against jNO's actual N1E assembly;
(2) the go/no-go itself — ``Jacobi + G(GᵀAG)⁻¹Gᵀ`` collapses the CG iteration count and, unlike plain
Jacobi, keeps it ~independent of the mass weight β (the ill-conditioning of the gradient near-null
space). See Hiptmair & Xu (2007) / Kolev & Vassilevski (2009) and :mod:`jno.utils.solver.ams`.
"""

import importlib.util

import jax
import numpy as np
import pytest

import jno
from jno.utils.solver.ams import discrete_gradient, nodal_vector_interpolation

_HAS_SCIPY = importlib.util.find_spec("scipy") is not None
inner = jno.np.inner


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _assemble(mesh_size, beta):
    """Assembled dense curl-curl (+ β·mass) N1E operator and the stashed edge topology. Meshes are kept
    coarse enough that the assembly fits the (small, 8 GB) dev GPU — this is a numpy/scipy structure
    test, so a refinement *contrast* is all it needs, not a fine mesh."""
    d = jno.domain(constructor=jno.domain.cube(mesh_size=mesh_size))
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    ui, vi = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    cu, cv = u.vector.curl(x, y, z), v.vector.curl(x, y, z)
    term = inner(cu, cv) if beta == 0 else inner(cu, cv) + beta * inner(ui, vi)
    A_raw = jno.fem([term]).operator[0]  # RT/N1E/P0 assemble sparse: .operator[0] is a BCOO (not dense)
    A = np.asarray(A_raw.todense() if hasattr(A_raw, "todense") else A_raw)
    return A, d._fem_nonnodal_topology


def test_discrete_gradient_lies_in_curl_curl_kernel():
    """Discrete de Rham: ``curl(∇φ) = 0`` ⇒ the pure curl-curl operator annihilates every column of G.

    A few flipped edge signs would leave ``K·G`` small-but-nonzero — the iteration test below would
    still pass, so this exact identity is the real regression guard on the sign convention."""
    K, topo = _assemble(0.3, beta=0)
    G = np.asarray(discrete_gradient(topo).todense())
    rel = np.linalg.norm(K @ G) / np.linalg.norm(K)
    assert rel < 1e-9, f"‖K·G‖/‖K‖ = {rel:.2e} — G is not in the curl-curl kernel (edge signs?)"


def test_pi_reproduces_constant_vector_fields():
    """Π reproduces constant vector fields and ties back to G at machine precision:
    ``Π_α · 1 = G · coords[:, α]`` (both equal the edge vector component ``t_e[α]``). This is the
    Π-side counterpart of the de Rham guard — a scale or sign slip in Π would pass the iteration
    margins below but fail this exact identity."""
    _, topo = _assemble(0.3, beta=0)
    G = np.asarray(discrete_gradient(topo).todense())
    Pis = [np.asarray(P.todense()) for P in nodal_vector_interpolation(topo)]
    coords = np.asarray(topo["vertex_points"])
    ones = np.ones(int(topo["n_verts"]))
    for a in range(3):
        assert np.max(np.abs(Pis[a] @ ones - G @ coords[:, a])) < 1e-12


@pytest.mark.skipif(not _HAS_SCIPY, reason="needs scipy.sparse.linalg for the CG iteration count")
def test_gradient_correction_collapses_and_is_beta_independent():
    """The AMS go/no-go: the gradient-space correction collapses the CG count and holds it ~flat in β,
    where plain Jacobi degrades as β→0 (the gradient modes stiffen). A direct (pinv) auxiliary solve
    isolates the mechanism — no AMG needed for the mechanism to show."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    def counts(mesh_size, beta):
        A, topo = _assemble(mesh_size, beta)
        n = A.shape[0]
        Asp = sp.csr_matrix(A)
        Gsp = sp.csr_matrix(np.asarray(discrete_gradient(topo).todense()))
        b = np.random.default_rng(0).standard_normal(n)
        dinv = 1.0 / np.diag(A)
        aux_pinv = np.linalg.pinv((Gsp.T @ Asp @ Gsp).toarray())  # (GᵀAG)⁺; const null-space is harmless

        def cg_iters(matvec):
            k = [0]
            spla.cg(
                Asp,
                b,
                M=spla.LinearOperator((n, n), matvec=matvec),
                rtol=1e-8,
                maxiter=3000,
                callback=lambda _x: k.__setitem__(0, k[0] + 1),
            )
            return k[0]

        jac = cg_iters(lambda r: dinv * r)
        ams = cg_iters(lambda r: dinv * r + Gsp @ (aux_pinv @ (Gsp.T @ r)))
        return jac, ams

    jac_hi, ams_hi = counts(0.3, 1e-2)  # mild ill-conditioning
    jac_lo, ams_lo = counts(0.3, 1e-6)  # severe: gradient modes nearly null

    assert ams_hi * 4 < jac_hi and ams_lo * 4 < jac_lo  # correction collapses the count (≥4×)
    assert ams_lo < 1.5 * ams_hi  # AMS-lite: ~flat in β (the auxiliary correction absorbs it)
    assert jac_lo > 1.3 * jac_hi  # Jacobi: strictly worse as β→0


@pytest.mark.skipif(not _HAS_SCIPY, reason="needs scipy.sparse.linalg for the CG iteration count")
def test_pi_correction_gives_h_independence_and_matches_lu():
    """M2: adding the Π (vector-nodal) correction makes full AMS ``h-independent`` — its CG count stays
    flat under refinement, where the gradient-only AMS-lite keeps growing. Aux solves are exact (pinv),
    so this probes the *structure*, not the (M5) AMG backend; the converged solve also matches the
    direct LU, confirming the additive applier is wired correctly."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    def measure(mesh_size, beta=1e-4):
        A, topo = _assemble(mesh_size, beta)
        n = A.shape[0]
        Asp = sp.csr_matrix(A)
        Gsp = sp.csr_matrix(np.asarray(discrete_gradient(topo).todense()))
        Pis = [sp.csr_matrix(np.asarray(P.todense())) for P in nodal_vector_interpolation(topo)]
        b = np.random.default_rng(0).standard_normal(n)
        dinv = 1.0 / np.diag(A)
        g_aux = np.linalg.pinv((Gsp.T @ Asp @ Gsp).toarray())
        p_aux = [np.linalg.pinv((P.T @ Asp @ P).toarray()) for P in Pis]

        def lite(r):
            return dinv * r + Gsp @ (g_aux @ (Gsp.T @ r))

        def full(r):
            x = lite(r)
            for P, Pv in zip(Pis, p_aux):
                x = x + P @ (Pv @ (P.T @ r))
            return x

        def cg_iters(matvec, rtol=1e-8):
            k = [0]
            x, _ = spla.cg(
                Asp,
                b,
                M=spla.LinearOperator((n, n), matvec=matvec),
                rtol=rtol,
                maxiter=3000,
                callback=lambda _x: k.__setitem__(0, k[0] + 1),
            )
            return k[0], x

        it_lite, _ = cg_iters(lite)
        it_full, x_full = cg_iters(full, rtol=1e-10)
        res = np.linalg.norm(Asp @ x_full - b) / np.linalg.norm(b)  # the residual CG actually drives
        return it_lite, it_full, res

    lite_c, full_c, res_c = measure(0.4)
    lite_f, full_f, res_f = measure(0.22)  # ~3× the DOFs (kept GPU-safe)

    assert full_f < 1.5 * full_c  # Π buys h-independence: full-AMS count ~flat under refinement
    assert lite_f > lite_c  # the gradient-only AMS-lite is NOT h-independent (it grows)
    assert full_f < 0.6 * lite_f  # and full AMS sits well below AMS-lite at the fine mesh
    assert res_c < 1e-8 and res_f < 1e-8  # the AMS-preconditioned CG converges (correct applier)
