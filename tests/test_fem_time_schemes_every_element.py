"""Every time scheme marches every element family.

One small linear transient per family -- heat on Lagrange (triangles, quads, tets, hexes, lines, vector,
coupled P2+P1), on Hermite and the cover space; the biharmonic heat flow on Morley and Argyris; an eddy-current
decay on 2-D and 3-D Nedelec; mixed Darcy on RT/P0; unsteady Stokes on Taylor-Hood -- marched by
``bdf()``, an adaptive SDIRK3 and the consistent-mass exponential, against a tight adaptive SDIRK3 of the
same semidiscrete system (so only the time error is measured). Plus the four bugs the sweep found:

* the Hermite/Argyris/Morley step took the Krylov default, whose noise pinned every adaptive controller at
  its smallest step on Argyris (NaN at any practical budget);
* a 1-D Hermite (beam) transient did not build, and a ``u_tt`` beam returned NaN;
* ``exponential()`` read its wall mask off the lumped row sum, which vanishes on P2 vertices, Hermite slopes
  and edge DOFs (refused as "pressure rows"), its Lanczos broke down or raised on small blocks, and its
  symmetric path silently mis-integrated a non-symmetric operator.

Run with x64 (the FEM assembly is float64).
"""

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np
import pytest

import jno  # noqa: E402

PI = np.pi
NSTEPS = 21
inner, grad, trace, sin, hessian, laplacian = (
    jno.np.inner,
    jno.np.grad,
    jno.np.trace,
    jno.np.sin,
    jno.np.hessian,
    jno.np.laplacian,
)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _vars(d):
    co = d.variable("interior", split=True)
    cb = d.variable("boundary", split=True)
    return co[:-1], co[-1], cb[:-1], d.variable("initial", split=True)


def _bind(sym, X, t):
    return sym.bind(**dict(zip("xyz", X)), t=t)


def _heat(d, **sym_kw):
    u, phi = d.fem_symbols(**sym_kw)
    X, t, Xb, ci = _vars(d)
    ui, vi = _bind(u, X, t), _bind(phi, X, t)
    u0 = 1.0
    for c in ci[: len(X)]:
        u0 = u0 * sin(PI * c)
    lap = ui.x * vi.x if len(X) == 1 else inner(grad(ui, X), grad(vi, X))
    return jno.fem([ui.t * vi + lap, u(*Xb) - 0.0, u(*ci[: len(X)]) - u0])


def _rect(T=0.05, h=0.3):
    return jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=h).domain(time=(0.0, T, NSTEPS))


def _box(T=0.03, h=0.4):
    return jno.shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=h).domain(time=(0.0, T, NSTEPS))


def _line(T=0.1, h=0.1):
    return jno.domain(constructor=jno.domain.line(mesh_size=h), time=(0.0, T, NSTEPS))


def _plate(space, form):
    d = _rect(T=2.5e-3)
    u, phi = d.fem_symbols(space=space)
    X, t, Xb, ci = _vars(d)
    ui, vi = _bind(u, X, t), _bind(phi, X, t)
    return jno.fem([ui.t * vi + form(ui, vi, X), u(*Xb) - 0.0, u(ci[0], ci[1]) - sin(PI * ci[0]) * sin(PI * ci[1])])


def _beam(second=False, T=5e-3, nsteps=NSTEPS, h=0.1):
    d = jno.domain(constructor=jno.domain.line(mesh_size=h), time=(0.0, T, nsteps))
    u, phi = d.fem_symbols(space="Hermite")
    X, t, Xb, ci = _vars(d)
    ui, vi = _bind(u, X, t), _bind(phi, X, t)
    acc = ui.tt if second else ui.t
    return d, jno.fem([acc * vi + laplacian(ui, X) * laplacian(vi, X), u(*Xb) - 0.0, u(ci[0]) - sin(PI * ci[0])])


def _vector_p2():
    d = _rect()
    u, phi = d.fem_symbols(value_shape=(2,), order=2)
    X, t, Xb, ci = _vars(d)
    ui, vi = _bind(u, X, t), _bind(phi, X, t)
    s = sin(PI * ci[0]) * sin(PI * ci[1])
    eu, ev = jno.np.symgrad(ui, X), jno.np.symgrad(vi, X)
    frm = inner(ui.t, vi) + 2.0 * inner(eu, ev, n_contract=2) + trace(eu) * trace(ev)
    return jno.fem([frm, u(*Xb)[0] - 0.0, u(*Xb)[1] - 0.0, u(ci[0], ci[1]) - jno.np.vector(s, 0.5 * s)])


def _coupled_p2_p1():
    d = _rect()
    u, v = d.fem_symbols(names=("u", "v"), order=2)
    w, q = d.fem_symbols(names=("w", "q"), order=1)
    X, t, Xb, ci = _vars(d)
    ui, vi, wi, qi = (_bind(s, X, t) for s in (u, v, w, q))
    s0 = sin(PI * ci[0]) * sin(PI * ci[1])
    return jno.fem(
        [
            ui.t * vi + inner(grad(ui, X), grad(vi, X)) - wi * vi,
            wi.t * qi + 0.5 * inner(grad(wi, X), grad(qi, X)) + ui * qi,
            u(*Xb) - 0.0,
            w(*Xb) - 0.0,
            u(ci[0], ci[1]) - s0,
            w(ci[0], ci[1]) - 0.5 * s0,
        ]
    )


def _n1e(dim):
    d = _rect(T=0.1, h=0.4) if dim == 2 else _box(T=0.1)
    u, v = d.fem_symbols(value_shape=(dim,), names=("u", "v"), space="N1E")
    X, t, _, ci = _vars(d)
    ui, vi = _bind(u, X, t), _bind(v, X, t)
    cu, cv = ui.vector.curl(*X), vi.vector.curl(*X)
    curlcurl = cu * cv if dim == 2 else inner(cu, cv)  # the 2-D curl is a scalar
    u0 = jno.np.vector(-ci[1], ci[0]) if dim == 2 else jno.np.vector(-ci[1], ci[0], 0.0 * ci[2])
    return jno.fem([inner(ui.t, vi) + curlcurl + inner(ui, vi), u(*ci[:dim]) - u0])


def _rt_p0():
    d = _rect()
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="RT")
    p, q = d.fem_symbols(names=("p", "q"), space="P0")
    X, t, _, ci = _vars(d)
    ui, vi, pi, qi = (_bind(s, X, t) for s in (u, v, p, q))
    icp = p(ci[0], ci[1]) - sin(PI * ci[0]) * sin(PI * ci[1])
    return jno.fem([inner(ui, vi) - pi * trace(grad(vi, X)), pi.t * qi + qi * trace(grad(ui, X)), icp], quad_degree=4)


def _stokes():
    d = _rect(T=0.02)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    X, t, Xb, ci = _vars(d)
    ui, vi, pi, qi = (_bind(s, X, t) for s in (u, v, p, q))
    x0, y0 = ci[0], ci[1]
    u0 = jno.np.vector(PI * sin(PI * x0) ** 2 * sin(2 * PI * y0), -PI * sin(2 * PI * x0) * sin(PI * y0) ** 2)
    frm = (
        inner(ui.t, vi) + inner(grad(ui, X), grad(vi, X), n_contract=2) - pi * trace(grad(vi, X)) - qi * trace(grad(ui, X))
    )
    return jno.fem([frm, u(*Xb)[0] - 0.0, u(*Xb)[1] - 0.0, u(x0, y0) - u0])


FAMILIES = {
    "tri_P1": lambda: _heat(_rect()),
    "tri_P2": lambda: _heat(_rect(), order=2),
    "tri_P3": lambda: _heat(_rect(h=0.4), order=3),
    "quad_Q2": lambda: _heat(
        jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=4).quad().domain(time=(0.0, 0.05, NSTEPS)), order=2
    ),
    "tet_P1": lambda: _heat(_box()),
    "hex_Q1": lambda: _heat(
        jno.shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).structured(n=3).quad().domain(time=(0.0, 0.03, NSTEPS))
    ),
    "line_P2": lambda: _heat(_line(h=0.2), order=2),
    "vector_P2": _vector_p2,
    "coupled_P2_P1": _coupled_p2_p1,
    "hermite_2d": lambda: _heat(_rect(), space="Hermite"),
    "cover": lambda: _heat(_rect(), space="cover"),
    "morley": lambda: _plate("Morley", lambda a, b, X: inner(hessian(a, X), hessian(b, X), n_contract=2)),
    "argyris": lambda: _plate("Argyris", lambda a, b, X: laplacian(a, X) * laplacian(b, X)),
    "hermite_beam": lambda: _beam()[1],
    "n1e_2d": lambda: _n1e(2),
    "n1e_3d": lambda: _n1e(3),
    "rt_p0_darcy": _rt_p0,
    "taylor_hood_stokes": _stokes,
}
GAUGE = {"taylor_hood_stokes": 1}  # the pressure block of a gauge-free Stokes: compared mean-free
# How the consistent-mass exponential is expected to do. Algebraic rows that are not walls (the RT flux, the
# Stokes pressure) are refused by design; a one-way coupling makes the operator non-symmetric (Arnoldi); the
# Argyris mass (condition near 1e9) caps its CG mass solve near 1e-3 (an open issue).
EXPONENTIAL = {
    "rt_p0_darcy": "refused",
    "taylor_hood_stokes": "refused",
    "coupled_P2_P1": ("symmetric=False", 1e-6),
    "argyris": ("consistent", 2e-3),
}
S = jno.solve


def _march(fem, scheme, save):
    out = fem.solve(save_ts=save, time=scheme)
    return np.array(out.fn() if hasattr(out, "fn") else jnp.asarray(out))


@pytest.mark.parametrize("family", list(FAMILIES))
def test_every_family_marches_on_bdf_sdirk3_and_the_exponential(family):
    """Each scheme agrees with the tight reference far below the gap between it and backward Euler on the
    domain grid (measured 2e-3 to 7e-2 on these forms), so a scheme that silently mis-integrates a family
    cannot pass."""
    fem = FAMILIES[family]()
    save = jnp.linspace(float(fem.t0), float(fem.t1), NSTEPS)[::4]
    gauge = GAUGE.get(family)

    def run(sch):
        x = _march(fem, sch, save)
        if gauge is not None:
            a, b = list(fem.offsets)[gauge : gauge + 2]
            x[:, a:b] -= x[:, a:b].mean(axis=1, keepdims=True)
        return x

    # the index-1 Darcy DAE takes the adaptive SDIRK3 many more steps: a looser, longer reference
    tight = (
        dict(rtol=1e-7, atol=1e-10, max_steps=20000)
        if family == "rt_p0_darcy"
        else dict(rtol=1e-8, atol=1e-11, max_steps=4000)
    )
    ref = run(S.sdirk(3).adaptive(**tight))
    assert np.all(np.isfinite(ref))
    scale = np.max(np.abs(ref))
    exp = EXPONENTIAL.get(family, ("consistent", 1e-6))
    cases = [
        ("bdf", S.bdf().adaptive(rtol=1e-5, atol=1e-8, max_steps=500), 2e-4),
        ("sdirk3 adaptive", S.sdirk(3).adaptive(rtol=1e-5, atol=1e-8, max_steps=500), 2e-4),
    ]
    if exp == "refused":
        with pytest.raises(NotImplementedError, match="algebraic"):
            run(S.exponential(mass="consistent"))
    else:
        cases.append(("exponential", S.exponential(mass="consistent", symmetric=exp[0] != "symmetric=False"), exp[1]))
    for label, sch, bound in cases:
        got = run(sch)
        assert np.all(np.isfinite(got)), f"{family}: {label} returned NaN"
        err = np.max(np.abs(got[1:] - ref[1:])) / scale
        assert err < bound, f"{family}: {label} is {err:.2e} from the reference"


@pytest.mark.parametrize("scheme", [S.bdf(), S.sdirk(3), S.theta(0.5)], ids=["bdf", "sdirk3", "theta05"])
def test_an_argyris_plate_marches_adaptively_within_a_small_budget(scheme):
    """The C1 families step on a sparse-direct solve. On the Krylov default every controller sat at its
    smallest step (all 10000 accepted steps at dt_min) and returned NaN at any budget below several thousand."""
    fem = FAMILIES["argyris"]()
    out = _march(fem, scheme.adaptive(rtol=1e-5, atol=1e-8, max_steps=250), jnp.asarray([float(fem.t1)]))
    assert np.all(np.isfinite(out))


def test_a_hermite_beam_decays_at_the_analytic_rate():
    """u_t + u_xxxx = 0, simply supported, u0 = sin(pi x): u = exp(-pi^4 t) sin(pi x). The initial state is
    the L2 projection, so its slope DOFs carry pi cos(pi x), not 0."""
    d, fem = _beam(T=5e-3, nsteps=41, h=0.05)
    xs = np.asarray(d.mesh.points)[:, 0]
    s0 = np.asarray(fem.state0).reshape(-1)
    order = np.argsort(xs)
    assert abs(s0[2 * order[0] + 1] - PI) < 1e-3  # slope at x = 0
    out = _march(fem, S.bdf().adaptive(rtol=1e-6, atol=1e-9), jnp.asarray([5e-3]))
    w = out[-1][0 : 2 * xs.size : 2]
    exact = np.exp(-(PI**4) * 5e-3) * np.sin(PI * xs)
    assert np.max(np.abs(w - exact)) / np.max(np.abs(exact)) < 1e-4


def test_a_vibrating_hermite_beam_keeps_its_amplitude():
    """u_tt + u_xxxx = 0, simply supported, at rest: u = cos(pi^2 t) sin(pi x), on the default (undamped,
    theta = 1/2) march. The Krylov step returned NaN here."""
    d, fem = _beam(second=True, T=0.1, nsteps=201, h=0.05)
    xs = np.asarray(d.mesh.points)[:, 0]
    out = fem.solve()
    out = np.asarray(out.fn() if hasattr(out, "fn") else out)
    w = out[-1][0 : 2 * xs.size : 2]
    exact = np.cos(PI**2 * 0.1) * np.sin(PI * xs)
    assert np.max(np.abs(w - exact)) / np.max(np.abs(exact)) < 1e-4


def test_lumped_exponential_runs_on_a_small_block_and_refuses_a_basis_it_cannot_lump():
    """A block with fewer unknowns than the Krylov size ran into "num_matvecs exceeds the acceptable
    range"; a P2 field, whose vertex row sums are 0, is refused by name rather than read as walls."""
    for fem in (FAMILIES["tri_P1"](), FAMILIES["tet_P1"]()):
        out = _march(fem, S.exponential(), jnp.asarray([0.0, float(fem.t1)]))
        assert np.all(np.isfinite(out))
    # the extreme: a mesh with no interior node, every DOF a wall -- the answer is 0, not 0/0
    walls_only = _heat(jno.shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=1.5).domain(time=(0.0, 0.03, 5)))
    for mass in ("lumped", "consistent"):
        out = _march(walls_only, S.exponential(mass=mass), jnp.asarray([0.0, 0.03]))
        assert np.all(np.isfinite(out)) and np.max(np.abs(out[-1])) < 1e-12
    with pytest.raises(NotImplementedError, match="mass='consistent'"):
        _march(FAMILIES["tri_P2"](), S.exponential(), jnp.asarray([0.0, 0.05]))


def test_symmetric_exponential_refuses_a_nonsymmetric_operator():
    """Lanczos on a non-symmetric operator returned an answer 1.4e-4 off, silently (coupled P2/P1 fields)."""
    with pytest.raises(NotImplementedError, match="symmetric=False"):
        _march(FAMILIES["coupled_P2_P1"](), S.exponential(mass="consistent"), jnp.asarray([0.0, 0.05]))
