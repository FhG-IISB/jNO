"""Guards for the integral-equation core (:mod:`jno.utils.solver.kernel`).

Every rung has an oracle that does not come from the code under test: a naive dense sum for the
pair operator, a dense Toeplitz build for the FFT operator, ``K x`` for the gradient, and Maxwell's
elliptic formula for the physics.

Two of these pin bugs that were found the expensive way and are invisible in the forward answer:

* the pair sum's diagonal must be excluded **by index**. Distances come from
  ``r² = |a|² + |b|² − 2a·b``, which cancels catastrophically on the diagonal — at coordinates ~100
  units from the origin it leaves ``r ≈ 3e-6`` rather than 0, so an epsilon threshold misses it and
  ``1/r`` injects an enormous spurious self term. ``test_pair_diagonal_survives_far_from_origin``
  places the elements exactly where that happens.
* the lattice generator's index wrap must be right on **every** axis. A cube hides a transposed
  axis, so ``test_lattice_matches_dense`` uses an anisotropic cell.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jno.utils.solver.kernel import lattice_operator, pair_matrix, pair_quadratic, sphere_self

INV_R = lambda r: 1.0 / r  # noqa: E731


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _dense_quadratic(pos, mom, self_g):
    """The definition, written out: no chunking, no scan, no distance identity."""
    pos, mom = np.asarray(pos), np.asarray(mom)
    if mom.ndim == 1:
        mom = mom[:, None]
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    mm = mom @ mom.T
    off = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
    return float((mm * off).sum() + (mom * mom).sum(1) @ np.asarray(self_g))


@pytest.mark.parametrize("shift", [0.0, 100.0])
def test_pair_diagonal_survives_far_from_origin(shift):
    """The pair sum matches the naive definition, including with the elements far from the origin.

    ``shift=100`` is the case that broke: ``r² = |a|² + |b|² − 2a·b`` has ~1e-11 of absolute
    cancellation error at that magnitude, so the diagonal comes out at ~3e-6 instead of 0.
    """
    key = jax.random.PRNGKey(0)
    pos = jax.random.uniform(key, (60, 3)) + shift
    mom = jax.random.normal(jax.random.PRNGKey(1), (60, 3))
    self_g = sphere_self(jnp.full((60,), 1e-3))
    got = float(pair_quadratic(pos, mom, INV_R, self_g, chunk=16))
    ref = _dense_quadratic(pos, mom, self_g)
    # 1e-10, not 1e-12: the reference sums in numpy and the operator sums on the accelerator, and a
    # GPU reassociates a 3600-term reduction differently (~4e-12 relative here). The defect this
    # guards is nothing like that size — a surviving 3e-6 diagonal puts a 1/r term of ~3e5 into a
    # total of ~4e3, i.e. off by a factor of 70 — so the bound is still twelve orders inside it.
    assert abs(got / ref - 1) < 1e-10, f"shift={shift}: {got:.12e} vs {ref:.12e}"


def test_pair_scalar_density_and_chunking_are_invariant():
    """A scalar density is accepted, and the answer does not depend on the chunk size."""
    pos = jax.random.uniform(jax.random.PRNGKey(2), (37, 3)) * 5.0
    mom = jax.random.normal(jax.random.PRNGKey(3), (37,))
    sg = jnp.full((37,), 2.0)
    vals = [float(pair_quadratic(pos, mom, INV_R, sg, chunk=c)) for c in (4, 8, 16, 64)]
    assert max(abs(v / vals[0] - 1) for v in vals) < 1e-12


def test_pair_requires_a_self_term():
    """A missing diagonal must raise, not fall back to a far-field value on a singular integral."""
    pos = jnp.zeros((4, 3)).at[:, 0].set(jnp.arange(4.0))
    with pytest.raises(ValueError, match="self_g is required"):
        pair_quadratic(pos, jnp.ones((4, 3)), INV_R, None)


def test_pair_is_differentiable_in_the_density():
    """``d/dm (m'Km) = 2Km``, so the gradient has a reference independent of finite differences."""
    pos = jax.random.uniform(jax.random.PRNGKey(4), (25, 3)) * 3.0
    sg = jnp.full((25,), 1.5)
    m0 = jax.random.normal(jax.random.PRNGKey(5), (25,))
    g = np.asarray(jax.grad(lambda m: pair_quadratic(pos, m, INV_R, sg))(m0))
    p = np.asarray(pos)
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    K = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), np.asarray(sg))
    assert np.abs(g - 2.0 * (K @ np.asarray(m0))).max() < 1e-9


def test_pair_is_differentiable_in_the_GEOMETRY():
    """The payoff: a gradient in the element POSITIONS, which is what mesh-free inverse design needs."""
    mom = jax.random.normal(jax.random.PRNGKey(6), (18, 3))
    sg = jnp.full((18,), 1.0)
    f = lambda p: pair_quadratic(p, mom, INV_R, sg)  # noqa: E731
    p0 = jax.random.uniform(jax.random.PRNGKey(7), (18, 3)) * 4.0 + 10.0
    ad = np.asarray(jax.grad(f)(p0))
    h, fd = 1e-6, np.zeros_like(ad)
    for i in (0, 5, 17):
        for k in range(3):
            fd[i, k] = (f(p0.at[i, k].add(h)) - f(p0.at[i, k].add(-h))) / (2 * h)
    for i in (0, 5, 17):
        assert np.abs(ad[i] / fd[i] - 1).max() < 1e-6


@pytest.mark.parametrize("n", [(4, 5, 3), (7, 6, 5)])
def test_lattice_matches_dense(n):
    """BTTB/FFT apply against a dense Toeplitz build, on an ANISOTROPIC cell.

    A cube would hide a transposed axis in the index wrap; unequal spacings do not.
    """
    h = (0.30, 0.22, 0.18)
    self_g = float(sphere_self(jnp.asarray(h[0] * h[1] * h[2])))
    apply = lattice_operator(n, h, INV_R, self_g)
    x = jax.random.normal(jax.random.PRNGKey(8), n)

    idx = np.stack(np.meshgrid(*[np.arange(v) for v in n], indexing="ij"), -1).reshape(-1, 3)
    pos = idx * np.array(h)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    K = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), self_g)
    ref = (K @ np.asarray(x).reshape(-1)).reshape(n)
    got = np.asarray(apply(x))
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-12


def test_lattice_is_differentiable():
    """``d/dx (½ x'Kx) = Kx``: the FFT apply carries a gradient, with an exact reference."""
    n, h = (5, 4, 3), (0.3, 0.25, 0.2)
    self_g = float(sphere_self(jnp.asarray(h[0] * h[1] * h[2])))
    apply = lattice_operator(n, h, INV_R, self_g)
    x0 = jax.random.normal(jax.random.PRNGKey(9), n)
    g = np.asarray(jax.grad(lambda x: 0.5 * jnp.sum(x * apply(x)))(x0))
    assert np.abs(g - np.asarray(apply(x0))).max() < 1e-10


def test_lattice_apply_is_jittable():
    """It has to survive ``jit`` — the point of the FFT path is many applies inside a solve."""
    n, h = (6, 5, 4), (0.2, 0.2, 0.2)
    apply = lattice_operator(n, h, INV_R, float(sphere_self(jnp.asarray(0.008))))
    x = jax.random.normal(jax.random.PRNGKey(10), n)
    assert np.allclose(np.asarray(jax.jit(apply)(x)), np.asarray(apply(x)), rtol=1e-12)


def test_mutual_inductance_of_two_coaxial_loops():
    """The physics oracle: Maxwell's elliptic formula, which is exact.

        M = mu0 sqrt(R1 R2) [ (2/k - k) K(k^2) - (2/k) E(k^2) ],  k^2 = 4 R1 R2/((R1+R2)^2 + d^2)

    ``pair_quadratic`` returns the FULL quadratic form, so the cross term is recovered as
    ``(Q_both - Q_1 - Q_2)/2`` — which also checks that within-block and cross-block pairs are
    summed consistently. Self terms are zero for filaments and cancel in the subtraction.
    """
    ellip = pytest.importorskip("scipy.special")
    mu0, R = 4e-7 * np.pi, 0.010

    def loop(z, N=256):
        th = np.arange(N) * 2 * np.pi / N + np.pi / N
        p = np.stack([R * np.cos(th), R * np.sin(th), np.full(N, z)], -1)
        t = np.stack([-R * np.sin(th), R * np.cos(th), np.zeros(N)], -1) * (2 * np.pi / N)
        return p, t

    for d in (0.005, 0.010, 0.030):
        p1, t1 = loop(0.0)
        p2, t2 = loop(d)
        Q = lambda p, t: float(  # noqa: E731
            pair_quadratic(jnp.asarray(p), jnp.asarray(t), INV_R, jnp.zeros(len(p)))
        )
        both = Q(np.vstack([p1, p2]), np.vstack([t1, t2]))
        cross = 0.5 * (both - Q(p1, t1) - Q(p2, t2))
        got = mu0 / (4 * np.pi) * cross

        k2 = 4 * R * R / ((2 * R) ** 2 + d**2)
        k = np.sqrt(k2)
        exact = mu0 * R * ((2 / k - k) * ellip.ellipk(k2) - (2 / k) * ellip.ellipe(k2))
        assert abs(got / exact - 1) < 2e-4, f"d={d}: {got * 1e9:.5f} nH vs {exact * 1e9:.5f} nH"


def test_group_none_equals_an_explicit_identity_labelling():
    """``group=None`` must be exactly ``group=arange(n)`` — the diagonal IS the same-element test.

    Pins the unification: there is one exclusion rule, not a diagonal branch plus a group branch.
    """
    pos = jax.random.uniform(jax.random.PRNGKey(11), (23, 3)) * 4.0
    mom = jax.random.normal(jax.random.PRNGKey(12), (23, 3))
    sg = jnp.full((23,), 0.7)
    a = float(pair_quadratic(pos, mom, INV_R, sg, chunk=8))
    b = float(pair_quadratic(pos, mom, INV_R, sg, group=np.arange(23), chunk=8))
    assert abs(a / b - 1) < 1e-13


def test_subpoint_quadrature_converges_on_a_straight_wire():
    """Partial inductances of N collinear elements must sum to the wire's closed-form self inductance.

        L = (mu0 l / 2pi) [ln(2l/a) - 3/4]

    Collinear neighbours are the worst case for a one-point mutual — they sit close relative to
    their own length — so this is the tightest available check on the near-field treatment. Measured
    at N = 32: 7.8 % low with one point per element, 2.5 % at two Gauss points, 0.2 % at eight.
    """
    from jno.utils.solver.kernel import wire_self

    mu0, L, A, N = 4e-7 * np.pi, 0.050, 2.5e-4, 32
    exact = (mu0 * L / (2 * np.pi)) * (np.log(2 * L / A) - 0.75)
    seg = L / N
    zc = (np.arange(N) + 0.5) * seg
    sg = jnp.asarray(np.full(N, float(wire_self(jnp.asarray(seg), A))))

    def total(nq):
        gx, gw = np.polynomial.legendre.leggauss(nq)
        z = (zc[:, None] + 0.5 * seg * gx[None, :]).ravel()
        w = np.tile(gw / 2.0, N)
        pos = jnp.asarray(np.stack([np.zeros_like(z), np.zeros_like(z), z], -1))
        mom = jnp.asarray(np.stack([np.zeros_like(z), np.zeros_like(z), seg * w], -1))
        q = pair_quadratic(pos, mom, INV_R, sg, group=np.repeat(np.arange(N), nq), chunk=64)
        return float(q) * mu0 / (4 * np.pi)

    err = [abs(total(q) / exact - 1) for q in (1, 2, 3, 5, 8)]
    assert err[0] > 0.05, f"a one-point mutual should be several percent low, got {err[0]:.4f}"
    assert all(b < a for a, b in zip(err, err[1:])), f"not monotone in quadrature order: {err}"
    assert err[-1] < 5e-3, f"eight Gauss points should be well under 1%, got {err[-1]:.4f}"


def test_single_element_reproduces_its_own_self_term():
    """With one element and no mutuals, the quadratic form IS the closed-form self inductance."""
    from jno.utils.solver.kernel import bar_self, wire_self

    mu0, L, A = 4e-7 * np.pi, 0.040, 2.0e-4
    pos = jnp.zeros((1, 3))
    mom = jnp.zeros((1, 3)).at[0, 2].set(L)
    got = float(pair_quadratic(pos, mom, INV_R, jnp.asarray([wire_self(L, A)]))) * mu0 / (4 * np.pi)
    assert abs(got / ((mu0 * L / (2 * np.pi)) * (np.log(2 * L / A) - 0.75)) - 1) < 1e-12

    w, t = 1.0e-3, 3.0e-4
    got = float(pair_quadratic(pos, mom, INV_R, jnp.asarray([bar_self(L, w, t)]))) * mu0 / (4 * np.pi)
    ref = (mu0 * L / (2 * np.pi)) * (np.log(2 * L / (w + t)) + 0.5 + 0.2235 * (w + t) / L)
    assert abs(got / ref - 1) < 1e-12


def test_pair_matrix_contracts_to_the_quadratic_form_under_any_current():
    """The matrix and the scalar path are separate implementations; x'Kx must reconcile them."""
    rng = np.random.default_rng(11)
    pos = jnp.asarray(rng.normal(size=(24, 3)))
    mom = jnp.asarray(rng.normal(size=(24, 3)))
    grp = jnp.asarray(np.repeat(np.arange(6), 4))
    sg = jnp.asarray(rng.uniform(0.5, 2.0, size=6))
    k = pair_matrix(pos, mom, INV_R, sg, group=grp)

    x = rng.normal(size=6)
    scaled = mom * jnp.asarray(x)[grp][:, None]
    # the self term is quadratic in the element moment, so it scales with x too
    q = float(pair_quadratic(pos, scaled, INV_R, sg, group=grp, chunk=5))
    assert abs(float(x @ np.asarray(k) @ x) / q - 1) < 1e-12
    assert np.allclose(np.asarray(k), np.asarray(k).T)


def test_pair_matrix_puts_the_self_term_on_the_diagonal():
    pos = jnp.asarray([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    mom = jnp.asarray([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    sg = jnp.asarray([0.7, 0.25])
    k = np.asarray(pair_matrix(pos, mom, INV_R, sg))
    assert np.allclose(np.diag(k), [0.7 * 4.0, 0.25 * 9.0])
    assert abs(k[0, 1] - 6.0 / 5.0) < 1e-13  # (m0 . m1) / r


def test_pair_matrix_requires_a_self_term():
    with pytest.raises(ValueError, match="self_g is required"):
        pair_matrix(jnp.zeros((3, 3)), jnp.ones((3, 3)), INV_R, None)
