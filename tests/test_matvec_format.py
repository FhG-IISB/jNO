"""Per-operator choice between cuSPARSE CSR and split COO for jNO's iterative solves.

Oracle: dense numpy for every product, JAX's own ``BCOO @`` for gradients, and the other format for the
end-to-end ``fem.solve()``.
"""

from __future__ import annotations

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

import jno
from jno.utils.solver import matvec_format as mf
from jno.utils.solver.linear import sparse_matvec


@pytest.fixture(autouse=True)
def _x64_and_restore():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        mf.set_matvec_format("auto")
        mf._DECISIONS.clear()
        jax.config.update("jax_enable_x64", prev)


@pytest.fixture
def logs(monkeypatch):
    """Messages sent to jNO's logger. Patched rather than captured from stdout: `jno.setup` (another
    test) re-binds the logger to a file logger whose console handler bypasses pytest's capture."""
    from jno.utils import logger as _logger

    got = []

    class _Spy:
        def info(self, m):
            got.append(m)

        warning = info

    monkeypatch.setattr(_logger, "get_logger", lambda *a, **k: _Spy())
    return got


def _ops(n=60, seed=0):
    """The operator shapes jNO produces: compressed (sorted, unique), uncompressed (unsorted duplicates),
    and padded (out-of-bound triplets from sum_duplicates)."""
    S = sp.random(n, n, density=0.08, random_state=seed, format="coo") + sp.eye(n)
    S = S.tocsr()
    S.sum_duplicates()
    S.sort_indices()
    compressed = jsp.BCOO.from_scipy_sparse(S.tocoo())
    coo = S.tocoo()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(coo.nnz)
    half = coo.data[perm] / 2.0  # every entry split into two unsorted duplicates
    idx = np.stack([coo.row[perm], coo.col[perm]], 1).astype(np.int32)
    uncompressed = jsp.BCOO(
        (jnp.asarray(np.concatenate([half, half])), jnp.asarray(np.concatenate([idx, idx]))), shape=S.shape
    )
    padded = compressed.sum_duplicates(nse=compressed.nse + 5)
    return S.toarray(), {"compressed": compressed, "uncompressed": uncompressed, "padded": padded}


@pytest.mark.parametrize("fmt", ["coo", "csr"])
def test_both_formats_reproduce_the_dense_product_on_every_operator_kind(fmt):
    mf.set_matvec_format(fmt)
    A, ops = _ops()
    rng = np.random.default_rng(1)
    v, V = rng.standard_normal(A.shape[1]), rng.standard_normal((A.shape[1], 3))
    for name, B in ops.items():
        np.testing.assert_allclose(
            np.asarray(sparse_matvec(B)(jnp.asarray(v))), A @ v, rtol=1e-12, atol=1e-12, err_msg=name
        )
        np.testing.assert_allclose(
            np.asarray(sparse_matvec(B)(jnp.asarray(V))), A @ V, rtol=1e-12, atol=1e-12, err_msg=name
        )
        np.testing.assert_allclose(
            np.asarray(sparse_matvec(B, transpose=True)(jnp.asarray(v))), A.T @ v, rtol=1e-12, atol=1e-12, err_msg=name
        )
        # traced operator (how every compiled solve receives it)
        got = jax.jit(lambda B, x: sparse_matvec(B)(x))(B, jnp.asarray(v))
        np.testing.assert_allclose(np.asarray(got), A @ v, rtol=1e-12, atol=1e-12, err_msg=name)


@pytest.mark.parametrize("fmt", ["coo", "csr"])
def test_both_formats_differentiate_and_vmap_like_bcoo(fmt):
    mf.set_matvec_format(fmt)
    _, ops = _ops()
    B = ops["uncompressed"]
    rng = np.random.default_rng(2)
    x, w = jnp.asarray(rng.standard_normal(B.shape[1])), jnp.asarray(rng.standard_normal(B.shape[0]))

    def L(fn):
        return lambda d, x: w @ fn(jsp.BCOO((d, B.indices), shape=B.shape))(x)

    got = jax.grad(L(sparse_matvec), argnums=(0, 1))(B.data, x)
    ref = jax.grad(L(lambda M: lambda v: M @ v), argnums=(0, 1))(B.data, x)
    for g, r in zip(got, ref):
        np.testing.assert_allclose(np.asarray(g), np.asarray(r), rtol=1e-12, atol=1e-12)
    Xs = jnp.asarray(rng.standard_normal((5, B.shape[1])))
    np.testing.assert_allclose(
        np.asarray(jax.vmap(sparse_matvec(B))(Xs)), np.asarray(jax.vmap(lambda v: B @ v)(Xs)), rtol=1e-12, atol=1e-12
    )


def test_auto_measures_once_per_operator_key_on_the_real_operator_when_concrete(logs):
    mf._DECISIONS.clear()
    _, ops = _ops(n=200)
    B = ops["compressed"]
    fmt = mf.choose(B)
    assert fmt in ("csr", "coo")
    assert len(logs) == 1 and "measured on the operator" in logs[0] and "->" in logs[0]
    assert mf.choose(B) == fmt and len(logs) == 1  # cached: no second timing
    (key,) = mf._DECISIONS
    dev = jax.devices()[0]
    assert key[:2] == (dev.platform, dev.device_kind)


def test_a_traced_operator_is_decided_on_a_banded_stand_in(logs):
    mf._DECISIONS.clear()
    _, ops = _ops(n=150, seed=3)
    B = ops["compressed"]
    jax.jit(lambda B, x: sparse_matvec(B)(x))(B, jnp.ones(B.shape[1]))
    assert any("banded stand-in" in m for m in logs)


def test_prime_decides_concrete_operators_and_ignores_the_rest(logs):
    mf._DECISIONS.clear()
    _, ops = _ops(n=120, seed=4)
    mf.prime(ops["compressed"], None, jnp.eye(3))
    assert len(mf._DECISIONS) == 1


def _solver_primitives(fmt):
    from jno._fem import _bicgstab_jacobi

    mf.set_matvec_format(fmt)
    S = sp.diags([-1.0, 2.2, -1.0], [-1, 0, 1], (40, 40)).tocoo()
    A = jsp.BCOO.from_scipy_sparse(S)
    return str(jax.make_jaxpr(lambda A, b: _bicgstab_jacobi(A, b, 1e-10, 500))(A, jnp.ones(40)))


def test_the_override_takes_effect_and_is_validated():
    assert "csr_matvec" in _solver_primitives("csr")
    txt = _solver_primitives("coo")
    assert "csr_matvec" not in txt and "scatter-add" in txt
    assert "csr_matvec" in _solver_primitives("csr")  # changing back takes effect (caches cleared)
    with pytest.raises(ValueError, match="matvec_format"):
        mf.set_matvec_format("bcsr")


def test_setup_sets_the_format_from_the_argument_and_from_toml(tmp_path, monkeypatch):
    from jno.utils import config

    script = tmp_path / "run.py"
    script.write_text("")
    jno.setup(str(script), matvec_format="csr")
    assert mf._FORMAT == "csr"
    monkeypatch.setattr(config, "get_config", lambda: {"jno": {"matvec_format": "coo"}})
    jno.setup(str(script))
    assert mf._FORMAT == "coo"


def test_fem_solve_gives_the_same_answer_with_either_format():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.05).domain()
    x, y = d.variable("interior", split=True)[:2]
    u, phi = d.fem_symbols()
    cb = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), phi.bind(x=x, y=y)
    f = jno.fn(lambda x, y: jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y), [x, y])
    sols = {}
    for fmt in ("coo", "csr"):
        mf.set_matvec_format(fmt)
        fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(cb[0], cb[1]) - 0.0])
        sols[fmt] = np.asarray(jax.tree_util.tree_leaves(fem.solve())[0])
    # two iterative solves converged to a 1e-8 relative residual: they agree to that order, not bitwise
    np.testing.assert_allclose(sols["csr"], sols["coo"], rtol=1e-6, atol=1e-10)
    assert np.abs(sols["coo"]).max() > 1e-3


def test_auto_measures_rectangular_operators_too(logs):
    """AMG prolongation / restriction operators are rectangular; the timing must not assume square."""
    mf._DECISIONS.clear()
    P = jsp.BCOO.from_scipy_sparse(sp.random(300, 40, density=0.05, random_state=7, format="coo"))
    assert mf.choose(P) in ("csr", "coo") and mf.choose(P.T) in ("csr", "coo")
    assert not any("could not time" in m for m in logs) and sum("per product" in m for m in logs) == 2


@pytest.mark.parametrize("fmt", ["coo", "csr"])
def test_amg_levels_in_the_measured_storage_are_the_same_v_cycle(fmt):
    """`build_hierarchy` stores each level's A/P/R in the chosen storage; the V-cycle must be the same
    linear map as the plain-BCOO one, and the levels a pytree of arrays."""
    from jno.utils.solver.amg import build_hierarchy, vcycle_apply

    mf.set_matvec_format(fmt)
    m = 30
    T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], (m, m))
    K = (sp.kron(T, sp.eye(m)) + sp.kron(sp.eye(m), T)).tocoo()  # 2-D Laplacian, 900 dofs
    levels = build_hierarchy(jsp.BCOO.from_scipy_sparse(K), coarse_size=20)
    assert all("fast" in lv and next(iter(lv["fast"]["A"])) == fmt for lv in levels if "A" in lv)
    plain = [{k: v for k, v in lv.items() if k != "fast"} for lv in levels]
    r = jnp.asarray(np.random.default_rng(8).standard_normal(K.shape[0]))
    np.testing.assert_allclose(
        np.asarray(jax.jit(lambda r: vcycle_apply(levels, r))(r)),
        np.asarray(vcycle_apply(plain, r)),
        rtol=1e-12,
        atol=1e-13,
    )
    leaves = jax.tree_util.tree_leaves(levels)
    assert all(hasattr(x, "shape") or isinstance(x, (int, float)) for x in leaves)


def test_operators_of_one_size_class_share_a_measurement():
    """An adaptive loop builds a new operator per remesh; re-measuring each cost ~0.5 s. Close sizes share
    the first one's decision, and a clearly different size is measured anew."""
    mf._DECISIONS.clear()
    a = jsp.BCOO.from_scipy_sparse(sp.diags([-1.0, 2.2, -1.0], [-1, 0, 1], (1000, 1000)).tocoo())
    b = jsp.BCOO.from_scipy_sparse(sp.diags([-1.0, 2.2, -1.0], [-1, 0, 1], (1100, 1100)).tocoo())
    c = jsp.BCOO.from_scipy_sparse(sp.diags([-1.0, 2.2, -1.0], [-1, 0, 1], (4000, 4000)).tocoo())
    mf.choose(a, log=False)
    mf.choose(b, log=False)
    assert len(mf._DECISIONS) == 1
    mf.choose(c, log=False)
    assert len(mf._DECISIONS) == 2
