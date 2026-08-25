"""``jno.solve.{logdet, trace, applyfun, diagonal, svd, lstsq}`` — matrix-free, differentiable matrix
functions and spectral quantities (via matfree).

``logdet`` / ``trace`` / ``diagonal`` are stochastic (Hutchinson + Lanczos quadrature) so gates are
loose; ``applyfun`` (``f(A)·v`` by Lanczos) is deterministic and essentially exact for the exponential.
One test trains a parameter *through* ``logdet`` — the Bayesian-evidence workflow the layer exists for.
``svd`` is the partial (Golub–Kahan) SVD of a rectangular operator, pinned against a dense oracle and
on the POD workflow that motivates it.
"""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from shapely.geometry import box

import jno
from jno.utils.solver.solver_api import LinearOperator

_HAS_MATFREE = importlib.util.find_spec("matfree") is not None
pytestmark = pytest.mark.skipif(not _HAS_MATFREE, reason="requires the optional 'matfree' package")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_krylov_breakdown_raises_instead_of_returning_nan():
    """The failure the suite's own fixture never saw, because it uses no Dirichlet term and a fine mesh.

    Lanczos can only build a subspace as large as the number of DISTINCT eigenvalues, and a pinned jNO
    FEM operator has far fewer than it has rows: every Dirichlet DOF is an identity row, so eigenvalue
    1.0 carries the pinned count as its multiplicity. Measured at mesh 0.25 -- n=30, 16 pinned rows,
    15 distinct eigenvalues -- the DEFAULT ``order=25`` overran that and ``logdet`` returned NaN while
    ``applyfun`` returned inf. Both must raise, and must do so under a trace too: these estimators
    exist to be differentiated, so an eager-only check would guard nothing."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.25)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    A, _ = jno.fem([ui * vi + ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0]).operator
    S = jnp.asarray(np.asarray(A.todense()))
    n = S.shape[0]
    ev = np.linalg.eigvalsh(np.asarray(S))
    assert len(np.unique(np.round(ev, 9))) < n, "the premise: pinning collapses the distinct spectrum"

    with pytest.raises(FloatingPointError, match="Krylov"):
        jno.solve.logdet(S, samples=8)  # default order=25 -> NaN before the guard
    with pytest.raises(Exception, match="Krylov"):
        jax.grad(lambda c: jno.solve.logdet(c * S, samples=8))(1.0)  # and under a trace

    # below the Krylov dimension it works, and stays differentiable
    exact = float(np.linalg.slogdet(np.asarray(S))[1])
    assert abs(float(jno.solve.logdet(S, samples=64, order=8)) - exact) / abs(exact) < 0.25
    assert np.isfinite(float(jax.grad(lambda c: jno.solve.logdet(c * S, samples=32, order=8))(1.0)))


def test_applyfun_krylov_overrun_is_still_only_partly_caught():
    """Records exactly how far the finiteness guard gets on ``applyfun`` -- which is NOT all the way.

    Same collapsed spectrum as above. ``applyfun`` degrades through three regimes as ``order`` passes
    the Krylov dimension, and only the last is caught:

        order <= 20   correct           (rel 1e-14 against a dense expm)
        order == 25   2.50e+37          FINITE, wrong by 35 orders -- NOT caught
        order == 29   9.05e+75          FINITE, wrong by 74 orders -- NOT caught
        order == 30   entries ~1e+300   the norm overflows to inf; caught only if an entry is inf

    A finiteness test cannot see a finite-but-absurd Krylov approximation. The real fix is to detect
    the Lanczos breakdown itself (beta ~ 0) and truncate to the invariant subspace, where the answer is
    EXACT -- which means computing f(T) from matfree's tridiagonal rather than using its
    ``funm_lanczos_sym`` composition. Not done; this test exists so the gap is recorded rather than
    rediscovered. Saad, *SIAM J. Numer. Anal.* 29(1), 1992, section 4 gives the standard error
    estimate for Krylov f(A)v approximations."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.25)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    A, _ = jno.fem([ui * vi + ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0]).operator
    S = jnp.asarray(np.asarray(A.todense()))
    n = S.shape[0]
    ones = jnp.ones(n)
    ref = np.linalg.norm(np.asarray(jax.scipy.linalg.expm(S)) @ np.ones(n))

    good = np.asarray(jno.solve.applyfun(S, ones, fun=jnp.exp, order=15))
    assert abs(np.linalg.norm(good) - ref) / ref < 1e-8, "inside the Krylov dimension it is exact"

    bad = np.asarray(jno.solve.applyfun(S, ones, fun=jnp.exp, order=25))
    assert np.all(np.isfinite(bad)), "the point: it is FINITE, so the guard cannot see it"
    assert np.max(np.abs(bad)) > 1e30, "and absurd -- 35 orders past the true answer"


def _spd_fem():
    """An SPD FEM operator (mass + stiffness) and its dense form."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.09)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    A = jno.fem([ui * vi + ui.x * vi.x + ui.y * vi.y]).operator[0]
    dense = jnp.asarray(A.todense() if hasattr(A, "todense") else A)
    return LinearOperator(A), dense


def test_logdet_matches_dense():
    op, dense = _spd_fem()
    est = float(jno.solve.logdet(op, samples=500, order=30, key=jax.random.PRNGKey(0)))
    true = float(jnp.linalg.slogdet(dense)[1])
    assert abs(est - true) / abs(true) < 0.02  # stochastic estimate


def test_trace_and_trace_of_inverse():
    op, dense = _spd_fem()
    tr = float(jno.solve.trace(op, samples=800, key=jax.random.PRNGKey(0)))
    assert abs(tr - float(jnp.trace(dense))) / float(jnp.trace(dense)) < 0.03
    tri = float(jno.solve.trace(op, fun=lambda z: 1.0 / z, samples=800, order=30, key=jax.random.PRNGKey(1)))
    assert abs(tri - float(jnp.trace(jnp.linalg.inv(dense)))) / float(jnp.trace(jnp.linalg.inv(dense))) < 0.05


def test_applyfun_matrix_exponential_is_exact():
    op, dense = _spd_fem()
    w = jnp.asarray(np.random.default_rng(0).standard_normal(dense.shape[0]))
    fv = jno.solve.applyfun(op, w, fun=lambda z: jnp.exp(-0.1 * z), order=35)
    true = jax.scipy.linalg.expm(-0.1 * dense) @ w
    assert float(jnp.linalg.norm(fv - true) / jnp.linalg.norm(true)) < 1e-8


def test_logdet_gradient_matches_analytic():
    """∂/∂s log det(sA) = n/s (=n at s=1) — the eigenvalue-scaling identity; autodiff flows through."""
    _op, dense = _spd_fem()
    n = dense.shape[0]
    key = jax.random.PRNGKey(0)

    def ld(s):
        return jno.solve.logdet(s * dense, samples=400, order=30, key=key)

    g = float(jax.grad(ld)(1.0))
    assert abs(g - n) / n < 0.02


@pytest.mark.slow
def test_logdet_trains_a_parameter():
    """Train a scale ``θ`` by matching ``logdet(θ·A)`` to a target — the differentiable-evidence loop.
    With a fixed key the estimator satisfies ``logdet(θA) = logdet(A) + n·log θ`` *exactly* (same probes),
    so the recovered optimum is θ*=2, and reaching it proves the gradient of ``logdet`` drives training."""
    _op, dense = _spd_fem()
    n = dense.shape[0]
    key = jax.random.PRNGKey(0)
    cfg = dict(samples=300, order=30, key=key)
    target = jno.solve.logdet(dense, **cfg) + n * jnp.log(2.0)  # ⇒ θ* = 2

    import optax

    def loss(theta):
        return (jno.solve.logdet(theta * dense, **cfg) - target) ** 2

    theta = jnp.array(1.0)
    opt = optax.adam(0.05)
    state = opt.init(theta)
    grad = jax.jit(jax.grad(loss))
    for _ in range(200):
        updates, state = opt.update(grad(theta), state)
        theta = optax.apply_updates(theta, updates)
    assert abs(float(theta) - 2.0) < 0.05  # recovered the scale by training through logdet


def _advection_fem():
    """A **non-symmetric** advection–diffusion FEM operator ``b·∇u + (∇u,∇v)`` and its dense form."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.16)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    A = jno.fem([5.0 * ui.x * vi + ui.x * vi.x + ui.y * vi.y]).operator[0]  # convection ⇒ A ≠ Aᵀ
    dense = jnp.asarray(A.todense() if hasattr(A, "todense") else A)
    return LinearOperator(A), dense


def test_applyfun_nonsymmetric_forward_is_exact():
    """``symmetric=False`` (Arnoldi + eig) computes ``f(A)·v`` for a non-symmetric advection–diffusion ``A``,
    forward-exact — the matrix exponential and a resolvent both match their dense references (GPU-capable)."""
    _op, dense = _advection_fem()
    w = jnp.asarray(np.random.default_rng(0).standard_normal(dense.shape[0]))
    fv = jno.solve.applyfun(dense, w, fun=lambda z: jnp.exp(-0.05 * z), order=40, symmetric=False)
    true = jax.scipy.linalg.expm(-0.05 * dense) @ w
    assert float(jnp.linalg.norm(fv - true) / jnp.linalg.norm(true)) < 1e-8

    fr = jno.solve.applyfun(dense, w, fun=lambda z: 1.0 / (3.0 + z), order=40, symmetric=False)
    truer = jnp.linalg.solve(3.0 * jnp.eye(dense.shape[0]) + dense, w)
    assert float(jnp.linalg.norm(fr - truer) / jnp.linalg.norm(truer)) < 1e-8


def test_applyfun_nonsymmetric_is_differentiable():
    """The non-symmetric path is **differentiable** for a general holomorphic ``fun`` — the Daleckii–Krein
    JVP supplies the matrix-function derivative analytically (no differentiating through ``eig``), so
    ``jax.grad`` flows and matches central finite differences to machine precision."""
    _op, dense = _advection_fem()
    w = jnp.asarray(np.random.default_rng(0).standard_normal(dense.shape[0]))

    def loss(s):
        return jnp.sum(jno.solve.applyfun(s * dense, w, fun=lambda z: jnp.exp(-0.05 * z), order=40, symmetric=False) ** 2)

    g = float(jax.grad(loss)(1.0))
    fd = float((loss(1.0 + 1e-6) - loss(1.0 - 1e-6)) / 2e-6)
    assert abs(g - fd) / abs(fd) < 1e-5


def test_diagonal_matches_dense():
    """``diagonal`` estimates the per-DOF diagonal field — of ``A`` and (the key use) of ``A⁻¹``, the
    pointwise variance map. Stochastic, so the whole-field error is loose."""
    op, dense = _spd_fem()
    d0 = jno.solve.diagonal(op, samples=4000, key=jax.random.PRNGKey(0))
    true0 = jnp.diagonal(dense)
    assert float(jnp.linalg.norm(d0 - true0) / jnp.linalg.norm(true0)) < 0.1
    # diag(A⁻¹) (the variance map) is a much higher-variance estimand — loose gate, honestly stochastic
    di = jno.solve.diagonal(op, fun=lambda z: 1.0 / z, samples=6000, order=30, key=jax.random.PRNGKey(1))
    truei = jnp.diagonal(jnp.linalg.inv(dense))
    assert float(jnp.linalg.norm(di - truei) / jnp.linalg.norm(truei)) < 0.2


def test_diagonal_gradient_matches_trace():
    """``∂/∂s Σ diag(sA) = Σ diag(A) = tr A`` — autodiff flows through the diagonal estimator."""
    _op, dense = _spd_fem()
    key = jax.random.PRNGKey(0)
    g = float(jax.grad(lambda s: jnp.sum(jno.solve.diagonal(s * dense, samples=2000, key=key)))(1.0))
    assert abs(g - float(jnp.trace(dense))) / float(jnp.trace(dense)) < 0.05


def test_lstsq_rectangular_and_damped():
    """LSMR least-squares for a rectangular ``A``: overdetermined matches ``np.linalg.lstsq``; ``damp``
    reproduces the Tikhonov normal equations ``(AᵀA + damp²I)x = Aᵀb``; and it differentiates in ``b``."""
    rng = np.random.default_rng(3)
    m, n = 120, 60
    A = jnp.asarray(rng.standard_normal((m, n)))
    b = jnp.asarray(rng.standard_normal(m))

    # LSMR is iterative (atol=btol=1e-6) so the gate is its convergence floor, not machine precision
    x = jno.solve.lstsq(A, b, atol=1e-10, btol=1e-10)
    xr = jnp.asarray(np.linalg.lstsq(np.asarray(A), np.asarray(b), rcond=None)[0])
    assert float(jnp.linalg.norm(x - xr) / jnp.linalg.norm(xr)) < 1e-5

    damp = 2.0
    xd = jno.solve.lstsq(A, b, damp=damp, atol=1e-10, btol=1e-10)
    xo = jnp.linalg.solve(A.T @ A + damp**2 * jnp.eye(n), A.T @ b)
    assert float(jnp.linalg.norm(xd - xo) / jnp.linalg.norm(xo)) < 1e-5

    gb = jax.grad(lambda bb: jnp.sum(jno.solve.lstsq(A, bb) ** 2))(b)
    assert bool(jnp.isfinite(gb).all())


def _decaying_svd_fixture(m=200, n=120, ratio=0.5, seed=1):
    """A rectangular operator with a geometrically decaying singular spectrum — the regime that makes
    POD and ill-posedness analysis meaningful (and the one Golub–Kahan converges on quickly)."""
    U0, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(seed), (m, m)))
    V0, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(seed + 1), (n, n)))
    s = jnp.asarray([10.0 * ratio**i for i in range(n)])
    return (U0[:, :n] * s) @ V0.T, s


def test_svd_recovers_the_leading_singular_triplets():
    """Partial SVD vs a dense oracle on a decaying spectrum, and the triplets must actually reconstruct
    the operator's dominant action — matching singular *values* alone would not catch swapped or
    misaligned vectors."""
    A, s_true = _decaying_svd_fixture()
    k = 6
    U, s, Vt = jno.solve.svd(A, k=k)

    assert U.shape == (A.shape[0], k) and s.shape == (k,) and Vt.shape == (k, A.shape[1])
    rel = float(jnp.max(jnp.abs(s - s_true[:k]) / s_true[:k]))
    assert rel < 1e-6, f"singular values off by {rel:.2e}"
    assert bool(jnp.all(jnp.diff(s) <= 0)), "singular values must come back descending"

    # the triplets reconstruct the optimal rank-k truncation (Eckart-Young-Mirsky): in the SPECTRAL
    # norm the residual is exactly the next singular value (in Frobenius it would be the tail norm
    # sqrt(sum_{i>k} s_i^2) instead -- a different statement)
    err2 = float(jnp.linalg.norm(A - (U * s) @ Vt, ord=2))
    assert abs(err2 - float(s_true[k])) / float(s_true[k]) < 1e-4, f"rank-{k} spectral residual {err2:.3e}"
    errF = float(jnp.linalg.norm(A - (U * s) @ Vt))
    tailF = float(jnp.sqrt(jnp.sum(s_true[k:] ** 2)))
    assert abs(errF - tailF) / tailF < 1e-4, f"rank-{k} Frobenius residual {errF:.3e} vs tail {tailF:.3e}"
    # orthonormal bases
    assert float(jnp.max(jnp.abs(U.T @ U - jnp.eye(k)))) < 1e-8
    assert float(jnp.max(jnp.abs(Vt @ Vt.T - jnp.eye(k)))) < 1e-8


def test_svd_is_matrix_free_and_differentiable():
    """The point of a *matrix-free* SVD: ``A`` is reached only through its matvec, so it can be the JVP
    of a differentiable solve rather than an assembled matrix — and the singular values differentiate
    back to whatever that matvec closes over (checked against a finite difference)."""
    A, _ = _decaying_svd_fixture(m=80, n=50, seed=7)
    op = LinearOperator.from_matvec(lambda v: A @ v, shape=A.shape)
    U, s, Vt = jno.solve.svd(op, k=4)
    s_dense = jnp.linalg.svd(A, compute_uv=False)[:4]
    assert float(jnp.max(jnp.abs(s - s_dense) / s_dense)) < 1e-6, "matvec-only path must match the dense SVD"

    # gradient of the leading singular value w.r.t. a scaling closed over by the matvec
    f = lambda c: jno.solve.svd(LinearOperator.from_matvec(lambda v: c * (A @ v), shape=A.shape), k=2)[1][0]  # noqa: E731
    g = float(jax.grad(f)(1.0))
    fd = float((f(1.0 + 1e-6) - f(1.0 - 1e-6)) / 2e-6)
    assert np.isfinite(g) and abs(g - fd) <= 1e-4 * max(abs(fd), 1.0), f"AD {g} vs FD {fd}"


def test_svd_depth_and_rank_limits_fail_loud():
    """Extremes. ``depth`` below ``k`` is the real footgun: the Ritz values converge from below, so at
    ``depth == k`` everything but the largest singular value is badly wrong — refuse it rather than
    return a plausible-looking spectrum."""
    A, s_true = _decaying_svd_fixture(m=60, n=40, seed=9)
    with pytest.raises(ValueError, match="depth"):
        jno.solve.svd(A, k=6, depth=3)
    for bad_k in (0, 41):
        with pytest.raises(ValueError, match="k="):
            jno.solve.svd(A, k=bad_k)

    # k=1 (the smallest useful request) and k=min(m,n) (the largest legal one) both work
    _U1, s1, _V1 = jno.solve.svd(A, k=1)
    assert abs(float(s1[0]) - float(s_true[0])) / float(s_true[0]) < 1e-6
    _Uf, sf, _Vf = jno.solve.svd(A, k=40)
    assert sf.shape == (40,) and bool(jnp.all(jnp.isfinite(sf)))

    # a wide operator (m < n) is as valid as a tall one
    Aw = A.T
    _Uw, sw, _Vw = jno.solve.svd(Aw, k=4)
    assert float(jnp.max(jnp.abs(sw - jnp.linalg.svd(Aw, compute_uv=False)[:4]) / sw)) < 1e-6


def test_svd_gives_a_pod_basis_from_a_fem_trajectory():
    """The motivating use: POD on a transient FEM trajectory.

    Two spatial modes with very different decay rates give a snapshot matrix of numerical rank ~2 — the
    fast mode is present at ``t=0`` and gone by the end. POD's actual claim is that a handful of modes
    captures the trajectory's energy, so that is what is pinned (two modes ≳99%), along with the leading
    right singular vector being the dominant *spatial* shape the trajectory relaxes onto."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.12, time=(0.0, 0.1, 21))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    # two modes with very different decay rates -> the fast one dies, leaving essentially one mode
    u0 = jno.np.sin(np.pi * ci[0]) * jno.np.sin(np.pi * ci[1]) + 0.8 * jno.np.sin(3 * np.pi * ci[0]) * jno.np.sin(
        3 * np.pi * ci[1]
    )
    fem = jno.fem([ui.t * vi + (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0])
    traj = np.asarray(fem.solve().fn())  # (n_time, n_dofs) snapshot matrix

    snaps = jnp.asarray(traj)
    k = 4
    _U, s, Vt = jno.solve.svd(snaps, k=k)
    s_ref = jnp.linalg.svd(snaps, compute_uv=False)[:k]
    assert float(jnp.max(jnp.abs(s - s_ref) / s_ref[0])) < 1e-6, "POD spectrum must match the dense SVD"
    # energy: the trajectory is numerically rank ~2 (one slow mode plus the fast one that dies early)
    total = float(jnp.sum(s_ref**2))
    energy1 = float(s[0] ** 2) / total
    energy2 = float(s[0] ** 2 + s[1] ** 2) / total
    assert energy1 > 0.9, f"the slow mode should dominate a diffused trajectory, got {energy1:.3f}"
    assert energy2 > 0.99, f"two POD modes should capture the trajectory, got {energy2:.3f}"
    # the POD basis is a *subspace*: the leading two right singular vectors reconstruct any frame,
    # including the last (projecting onto the single leading mode is a strictly weaker basis and does
    # not, since each frame is a different mixture of the two spatial modes)
    basis = Vt[:2]
    for frame in (snaps[0], snaps[len(snaps) // 2], snaps[-1]):
        proj = basis.T @ (basis @ frame)
        rel = float(jnp.linalg.norm(frame - proj) / jnp.linalg.norm(frame))
        assert rel < 0.05, f"the rank-2 POD basis must reconstruct every frame, got {rel:.3f}"


@pytest.mark.skipif(_HAS_MATFREE, reason="matfree installed; this checks the missing-dependency message")
def test_missing_matfree_message():
    _op, dense = _spd_fem()
    with pytest.raises(ImportError, match="matfree"):
        jno.solve.logdet(dense)
