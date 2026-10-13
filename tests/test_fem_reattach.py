"""A property re-attached after `jno.fem(...)` is built reaches the next `fem.solve()`.

`d.q` used to copy the attached values into the term when it was written, and the built `fem` kept its
assembly, so the second solve below silently returned the first answer. Now the built `fem` re-assembles
when a property it reads has changed since its last assembly, and says so at INFO.

Oracles: linearity in the source (the answer doubles exactly), and a fresh `jno.fem(...)` built after the
change (the reference the stale path used to disagree with).
"""

import numpy as np
import pytest

import jno


def _two_regions(qa=1.0, qb=0.0, ka=1.0):
    a = jno.shape.rect(0.0, 0.0, 0.5, 1.0).name("a").attach(q=qa, k=ka)
    b = jno.shape.rect(0.5, 0.0, 1.0, 1.0).name("b").attach(k=1.0, **({} if qb is None else {"q": qb}))
    d = (a + b).sized(0.125).domain()
    u, v = d.fem_symbols()
    x, y, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    xb, yb, _ = d.variable("boundary", split=True)
    return d, u, ui, vi, (xb, yb)


def _poisson(d, u, ui, vi, wall):
    return [d.k * (ui.x * vi.x + ui.y * vi.y) - d.q * vi, u(*wall) - 0.0]


def test_reattached_source_reaches_the_built_fem(capfd):
    d, u, ui, vi, wall = _two_regions()
    fem = jno.fem(_poisson(d, u, ui, vi, wall))
    u1 = np.asarray(fem.solve())
    capfd.readouterr()

    d.attach("a", q=2.0)
    u2 = np.asarray(fem.solve())
    np.testing.assert_allclose(u2, 2.0 * u1, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(u2, np.asarray(jno.fem(_poisson(d, u, ui, vi, wall)).solve()), rtol=1e-6, atol=1e-12)
    out = "".join(capfd.readouterr())
    assert "re-assembled" in out and "'q'" in out


def test_no_change_means_no_reassembly(capfd):
    d, u, ui, vi, wall = _two_regions()
    fem = jno.fem(_poisson(d, u, ui, vi, wall))
    fem.solve()
    capfd.readouterr()
    fem.solve()
    assert "re-assembled" not in "".join(capfd.readouterr())


def test_a_default_counts_as_a_change():
    """`d.attach(q=...)` (no target) sets the value for regions without their own; here region b."""
    d, u, ui, vi, wall = _two_regions(qa=1.0, qb=None)  # b declares no q of its own
    d.attach(q=1.0)
    fem = jno.fem(_poisson(d, u, ui, vi, wall))
    u1 = np.asarray(fem.solve())
    d.attach(q=3.0)  # b: 1 -> 3; a keeps its own 1.0
    u2 = np.asarray(fem.solve())
    np.testing.assert_allclose(u2, np.asarray(jno.fem(_poisson(d, u, ui, vi, wall)).solve()), rtol=1e-6, atol=1e-12)
    assert not np.allclose(u1, u2)


def test_reattached_conductivity_in_a_nonlinear_solve():
    """-div(k grad u) + u^3 = q: Newton, with region a's k changed between solves."""
    d, u, ui, vi, wall = _two_regions(qa=10.0, qb=10.0)
    terms = lambda: [d.k * (ui.x * vi.x + ui.y * vi.y) + ui**3 * vi - d.q * vi, u(*wall) - 0.0]  # noqa: E731
    newton = jno.solve.newton(rtol=0.0, atol=1e-5)  # reachable in single precision too
    fem = jno.fem(terms())
    u1 = np.asarray(fem.solve(nonlinear=newton))
    d.attach("a", k=5.0)
    u2 = np.asarray(fem.solve(nonlinear=newton))
    np.testing.assert_allclose(u2, np.asarray(jno.fem(terms()).solve(nonlinear=newton)), rtol=1e-5, atol=1e-8)
    assert np.abs(u2 - u1).max() > 1e-3


def test_reattaching_an_expression_value_refuses():
    """A value that is itself an expression is captured when the term is written; it cannot be
    re-read, so a later change must stop the solve rather than be ignored."""
    d, u, ui, vi, wall = _two_regions()
    d.attach("a", q=lambda x, y: 1.0 + x)
    fem = jno.fem(_poisson(d, u, ui, vi, wall))
    fem.solve()
    d.attach("a", q=lambda x, y: 2.0 + x)
    with pytest.raises(ValueError, match=r"'q'.*re-attached.*jno\.fem"):
        fem.solve()
