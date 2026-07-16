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
from jno.utils.solver.ams import discrete_gradient

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


def _curlcurl(mesh_size, beta):
    """Assembled dense curl-curl (+ β·mass) N1E operator and its discrete gradient G."""
    d = jno.domain(constructor=jno.domain.cube(mesh_size=mesh_size))
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    ui, vi = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    cu, cv = u.vector.curl(x, y, z), v.vector.curl(x, y, z)
    term = inner(cu, cv) if beta == 0 else inner(cu, cv) + beta * inner(ui, vi)
    fem = jno.fem([term])
    A = np.asarray(fem.operator[0])  # small mesh → assembled dense
    G = discrete_gradient(d._fem_nonnodal_topology)
    return A, np.asarray(G.todense())


def test_discrete_gradient_lies_in_curl_curl_kernel():
    """Discrete de Rham: ``curl(∇φ) = 0`` ⇒ the pure curl-curl operator annihilates every column of G.

    A few flipped edge signs would leave ``K·G`` small-but-nonzero — the iteration test below would
    still pass, so this exact identity is the real regression guard on the sign convention."""
    K, G = _curlcurl(0.3, beta=0)
    rel = np.linalg.norm(K @ G) / np.linalg.norm(K)
    assert rel < 1e-9, f"‖K·G‖/‖K‖ = {rel:.2e} — G is not in the curl-curl kernel (edge signs?)"


@pytest.mark.skipif(not _HAS_SCIPY, reason="needs scipy.sparse.linalg for the CG iteration count")
def test_gradient_correction_collapses_and_is_beta_independent():
    """The AMS go/no-go: the gradient-space correction collapses the CG count and holds it ~flat in β,
    where plain Jacobi degrades as β→0 (the gradient modes stiffen). A direct (pinv) auxiliary solve
    isolates the mechanism — no AMG needed for the mechanism to show."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    def counts(mesh_size, beta):
        A, G = _curlcurl(mesh_size, beta)
        n = A.shape[0]
        Asp, Gsp = sp.csr_matrix(A), sp.csr_matrix(G)
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
