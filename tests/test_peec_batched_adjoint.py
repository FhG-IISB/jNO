"""Several cotangents through ONE PEEC solve: `jax.vmap` over its vjp, and so `jax.jacrev`.

A design loop with constraints needs the gradient of several readouts of the same solve -- a loop
inductance and the current split between paralleled dies, say. Each is one adjoint solve against the
same transposed system, so they belong in ONE batched solve. Through jNO that raised
`NotImplementedError ... pure_callback ... vmap_method=None`: the host preconditioner is a
`pure_callback` with no batching rule. It now takes the batch as extra right-hand sides of the same
factorisation -- one host call per GMRES iteration for all of them.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
import jno.utils.solver.peec as PS

MM, SIG = 1e-3, 5.8e7
A1 = [(4 * MM, 2 * MM, 0.5 * MM), (4 * MM, 2 * MM, 1.5 * MM), (16 * MM, 2 * MM, 1.5 * MM), (16 * MM, 2 * MM, 0.5 * MM)]
A2 = [(4 * MM, 4 * MM, 0.5 * MM), (4 * MM, 4 * MM, 1.5 * MM), (16 * MM, 4 * MM, 1.5 * MM), (16 * MM, 4 * MM, 0.5 * MM)]


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _network():
    trace = jno.shape.box(0, 0, 0, 20 * MM, 6 * MM, 0.5 * MM, size=(1 * MM, 1 * MM, 0.5 * MM)).attach(sigma=SIG).name("t")
    w1 = jno.shape.line(A1, r=0.15 * MM, size=1 * MM).attach(sigma=SIG).name("w1")
    w2 = jno.shape.line(A2, r=0.15 * MM, size=1 * MM).attach(sigma=SIG).name("w2")
    d = (trace + w1 + w2).domain()
    d.tag("A", lambda x, y, z: x < 1.01 * MM)
    d.tag("B", lambda x, y, z: x > 18.99 * MM)
    _i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]  # noqa: E731
    return jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e6).build()


def _readouts(b):
    def F(h):
        geo = {"w1": (jnp.asarray(A1).at[1:3, 2].set(h[0]), None), "w2": (jnp.asarray(A2).at[1:3, 2].set(h[1]), None)}
        s = b.solve(geometry=geo)
        return jnp.stack([jnp.real(s.L) * 1e9, jnp.real(s.Z) * 1e3, jnp.imag(s.Z) * 1e3])

    return F


def test_batched_cotangents_equal_separate_ones_in_one_host_call_per_iteration():
    F = _readouts(_network())
    _out, vjp = jax.vjp(F, jnp.array([1.5 * MM, 1.4 * MM]))
    c0 = PS._LU_HOLDER.get("applies", 0)
    rows = np.stack([np.asarray(vjp(e)[0]) for e in jnp.eye(3)])
    separate = PS._LU_HOLDER.get("applies", 0) - c0
    c1 = PS._LU_HOLDER.get("applies", 0)
    J = np.asarray(jax.vmap(lambda e: vjp(e)[0])(jnp.eye(3)))
    batched = PS._LU_HOLDER.get("applies", 0) - c1
    assert np.abs(J - rows).max() <= 1e-6 * np.abs(rows).max()
    # one host call serves all three right-hand sides; the batch runs until the slowest converges
    assert batched < 0.6 * separate, (batched, separate)


def test_jacrev_through_a_solve():
    F = _readouts(_network())
    h = jnp.array([1.5 * MM, 1.4 * MM])
    J = np.asarray(jax.jacrev(F)(h))
    e = 1e-6
    fd = np.stack([(np.asarray(F(h.at[k].add(e))) - np.asarray(F(h.at[k].add(-e)))) / (2 * e) for k in range(2)], 1)
    assert np.allclose(J, fd, rtol=1e-4, atol=1e-6 * np.abs(fd).max())
