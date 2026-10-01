"""A state-dependent mass ``c(u) u_t`` marches under periodic ties.

A periodic march carries the REDUCED state. The state-dependent mass reads the previous step's fields off
it, sliced with the FULL layout's offsets, and its mass action was never reduced at all: a periodic
Navier-Stokes march with ``u_t`` inside its stabilisation (residual-based VMS) raised on its first step,
first on the slicing and then on adding a full-size mass action to a reduced residual.

Oracles: two decoupled fields with their own ``c(u) u_t`` and different uniform initial values follow their
own scalar recursion -- backward Euler or BDF2's non-conservative form, solved exactly on the host -- on
every node (a field sliced with the wrong offsets would read the other's values); and the exact 2-D
Taylor-Green field in a periodic box is marched with ``u_t`` inside SUPG/PSPG no worse than without.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


E = 1e-9
DT, NST = 0.1, 5
A0, B0, KA, KB = 0.3, -0.5, 1.0, 2.0


def _periodic_square(steps=NST):
    d = jno.shape.rect(0, 0, 1, 1).structured(n=4).domain(time=(0.0, DT * steps, steps + 1))
    for nm, f in {"l": lambda x, y: x < E, "r": lambda x, y: x > 1 - E,
                  "b": lambda x, y: y < E, "t": lambda x, y: y > 1 - E}.items():  # fmt: skip
        d.tag(nm, f)
    return d


def _recursion(u0, k, scheme, steps=NST):
    """Uniform field: (1 + w²)·rate + k w = 0, rate the scheme's own difference quotient."""

    def solve(rate_of, guess):
        w = guess
        for _ in range(60):  # scalar Newton on the cubic
            h = 1e-7
            f = (1 + w * w) * rate_of(w) + k * w
            df = ((1 + (w + h) ** 2) * rate_of(w + h) + k * (w + h) - f) / h
            w = w - f / df
        return w

    out = [u0]
    for n in range(steps):
        u = out[-1]
        if scheme == "be" or n == 0:  # BDF2 starts with one backward-Euler step
            out.append(solve(lambda w, u=u: (w - u) / DT, u))
        else:
            um = out[-2]
            out.append(solve(lambda w, u=u, um=um: (3 * w - 4 * u + um) / (2 * DT), u))
    return np.array(out)


@pytest.mark.parametrize("scheme", ["be", "bdf2"])
def test_two_fields_with_their_own_state_dependent_mass_march_on_a_periodic_square(scheme):
    d = _periodic_square()
    a, phi = d.fem_symbols(names=("a", "phi"))
    b, psi = d.fem_symbols(names=("b", "psi"))
    x, y, t = d.variable("interior", split=True)
    ai, pi = a.bind(x=x, y=y, t=t), phi.bind(x=x, y=y, t=t)
    bi, si = b.bind(x=x, y=y, t=t), psi.bind(x=x, y=y, t=t)
    at = lambda r: d.variable(r, split=True)[:2]  # noqa: E731
    c = d.variable("initial", split=True)
    terms = [
        (1 + ai * ai) * ai.t * pi + KA * ai * pi,
        (1 + bi * bi) * bi.t * si + KB * bi * si,
        a(*at("l")) - a(*at("r")), a(*at("b")) - a(*at("t")),
        b(*at("l")) - b(*at("r")), b(*at("b")) - b(*at("t")),
        a(*c) - A0, b(*c) - B0,
    ]  # fmt: skip
    fem = jno.fem(terms)
    kw = {"time": jno.solve.bdf2()} if scheme == "bdf2" else {}
    traj = np.asarray(fem.solve(**kw).fn())
    ga, gb = traj[:, fem.blocks[0]], traj[:, fem.blocks[1]]
    want_a, want_b = _recursion(A0, KA, scheme), _recursion(B0, KB, scheme)
    assert np.abs(want_a[-1] - A0) > 0.05 and np.abs(want_b[-1] - B0) > 0.05  # both actually decay
    assert np.abs(ga - want_a[:, None]).max() < 1e-9, "field a left its own recursion"
    assert np.abs(gb - want_b[:, None]).max() < 1e-9, "field b left its own recursion"


def test_a_vector_field_with_a_state_dependent_mass_marches_on_a_periodic_square():
    """(1 + |u|²) u_t + u = 0, uniform: each component decays by the SAME scalar factor per step."""
    d = _periodic_square()
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"))
    x, y, t = d.variable("interior", split=True)
    ui, vi = u.bind(x=x, y=y, t=t), v.bind(x=x, y=y, t=t)
    dot = lambda p, q: jno.np.inner(p, q, n_contract=1)  # noqa: E731
    at = lambda r: d.variable(r, split=True)[:2]  # noqa: E731
    c = d.variable("initial", split=True)
    u0 = np.array([0.3, -0.4])
    fem = jno.fem(
        [
            (1 + dot(ui, ui)) * dot(ui.t, vi) + dot(ui, vi),
            u(*at("l")) - u(*at("r")), u(*at("b")) - u(*at("t")),
            u(*c)[0] - u0[0], u(*c)[1] - u0[1],
        ]
    )  # fmt: skip
    traj = np.asarray(fem.solve().fn()).reshape(NST + 1, -1, 2)
    # the uniform state stays parallel to u0: w = s u0, (1 + s²|u0|²)(s - s_prev)/dt + s = 0
    want = [1.0]
    for _ in range(NST):
        sp, s = want[-1], want[-1]
        for _ in range(60):
            f = (1 + s * s * (u0 @ u0)) * (s - sp) / DT + s
            df = (2 * s * (u0 @ u0)) * (s - sp) / DT + (1 + s * s * (u0 @ u0)) / DT + 1
            s = s - f / df
        want.append(s)
    want = np.array(want)[:, None, None] * u0[None, None, :]
    assert np.abs(traj - want).max() < 1e-9


def test_u_t_inside_a_stabilised_periodic_navier_stokes_march():
    """The exact 2-D Taylor-Green field in a periodic box: its strong residual vanishes, so u_t inside
    SUPG/PSPG (residual-based VMS) must march it no worse than the quasi-static residual does."""
    grad, trace, inner, lap, sin, cos = (jno.np.grad, jno.np.trace, jno.np.inner, jno.np.laplacian,
                                         jno.np.sin, jno.np.cos)  # fmt: skip
    dot = lambda p, q: inner(p, q, n_contract=1)  # noqa: E731
    ddot = lambda p, q: inner(p, q, n_contract=2)  # noqa: E731
    L, nu, dt, nst = 2 * np.pi, 0.01, 0.1, 5

    def march(with_rate):
        d = jno.shape.box(0, 0, 0, L, L, L).structured(n=6).domain(time=(0.0, dt * nst, nst + 1))
        faces = {"x0": lambda x, y, z: x < E, "x1": lambda x, y, z: x > L - E,
                 "y0": lambda x, y, z: y < E, "y1": lambda x, y, z: y > L - E,
                 "z0": lambda x, y, z: z < E, "z1": lambda x, y, z: z > L - E}  # fmt: skip
        for nm, f in faces.items():
            d.tag(nm, f)
        d.point_region("pin", (np.pi, np.pi, np.pi))
        u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"))
        p, q = d.fem_symbols(names=("p", "q"))
        x, y, z, t = d.variable("interior", split=True)
        X = [x, y, z]
        ui, vi = u.bind(x=x, y=y, z=z, t=t), v.bind(x=x, y=y, z=z, t=t)
        pi, qi = p.bind(x=x, y=y, z=z, t=t), q.bind(x=x, y=y, z=z, t=t)
        gu, gv, gp, gq = grad(u, X), grad(v, X), grad(p, X), grad(q, X)
        G = d.cell_metric
        tau = ((2.0 / dt) ** 2 + dot(ui, dot(G, ui)) + 36.0 * nu**2 * ddot(G, G)) ** -0.5
        R = dot(gu, ui) - nu * lap(u, X) + gp
        terms = [
            dot(ui.t, vi) + dot(dot(gu, ui), vi) + nu * ddot(gu, gv) - pi * trace(gv), qi * trace(gu),
            tau * dot(dot(gv, ui), R), tau * dot(gq, R),
        ]  # fmt: skip
        if with_rate:
            terms += [tau * dot(dot(gv, ui), ui.t), tau * dot(gq, ui.t)]
        at = lambda r: d.variable(r, split=True)[:3]  # noqa: E731
        terms.append(p(*at("pin")) - 0.0)
        for lo, hi in (("x0", "x1"), ("y0", "y1"), ("z0", "z1")):
            terms += [u(*at(lo)) - u(*at(hi)), p(*at(lo)) - p(*at(hi))]
        c = d.variable("initial", split=True)
        terms += [u(*c)[0] - sin(c[0]) * cos(c[1]), u(*c)[1] - (-cos(c[0]) * sin(c[1])), u(*c)[2] - 0.0]
        fem = jno.fem(terms)
        traj = np.asarray(fem.solve(time=jno.solve.bdf2(), save_ts=np.array([0.0, dt * nst])).fn())
        pts = np.asarray(fem.points)
        U = traj[-1][fem.blocks[fem.block_index(u)]].reshape(len(pts), 3)
        k = np.exp(-2 * nu * dt * nst)
        ex = np.stack([k * np.sin(pts[:, 0]) * np.cos(pts[:, 1]), -k * np.cos(pts[:, 0]) * np.sin(pts[:, 1]),
                       0 * pts[:, 0]], 1)  # fmt: skip
        return np.linalg.norm(U - ex) / np.linalg.norm(ex)

    quasi_static, consistent = march(False), march(True)
    assert np.isfinite(consistent) and consistent < 0.2
    assert consistent <= 1.05 * quasi_static, (consistent, quasi_static)
