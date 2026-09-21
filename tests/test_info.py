"""`jno.info` — one front door for what an object is.

It replaced three spellings on three classes (`domain.summary()`, `core.print_tree()`,
`core.print_shapes()`), so the first thing these tests pin is that the content those carried is
still reachable, and the second is that the cheap/deep split is real.
"""

from __future__ import annotations

import importlib.util

import jax
import numpy as np
import pytest

import jno


def _two_region_domain():
    lo = jno.shape.rect(0, 0, 2, 1).name("lo").attach(k=5.0)
    hi = jno.shape.rect(0, 1, 2, 2).name("hi").attach(k=1.0)
    d = (lo + hi).sized(0.4).domain()
    d.tag("lid", lambda x, y: y > 2 - 1e-9)
    return d


def _poisson(d):
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u, v = d.fem_symbols()
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    return jno.fem([d.k * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0])


def test_domain_info_reports_mesh_regions_and_attachments():
    d = _two_region_domain()
    rep = jno.info(d)
    text, data = str(rep), rep.as_dict()
    assert "domain" in text and "2D" in text
    assert data["mesh"]["built"] == "yes"
    assert "triangle" in data["mesh"] and "worst aspect" in data["mesh"]["triangle"]
    # the attachment view is the thing `by_region`'s retirement made load-bearing
    assert "d.k" in data["attached (d.<prop>)"]
    assert "lo=5.0" in data["attached (d.<prop>)"]["d.k"]


def test_domain_info_says_when_the_mesh_is_still_lazy():
    d = jno.shape.rect(0, 0, 1, 1, size=0.5).domain()
    assert "lazy" in jno.info(d).as_dict()["mesh"]["built"]


def test_fem_info_reports_mode_dofs_and_every_term():
    f = _poisson(_two_region_domain())
    data = jno.info(f).as_dict()
    assert data["form"]["mode"] == "linear"
    assert int(str(data["form"]["dofs"]).replace(",", "")) == int(f.dofs)
    # every term, classified -- the check that a boundary condition was recognised AS one
    assert list(data["terms (as classified)"].values()) == list(f.classification)


def test_fem_info_deep_adds_symmetry_and_empty_rows():
    f = _poisson(_two_region_domain())
    cheap, deep = jno.info(f).as_dict()["operator"], jno.info(f, deep=True).as_dict()["operator"]
    assert "symmetry" not in cheap, "the dense check must not run unless asked"
    assert "symmetric" in deep["symmetry"] and deep["empty rows"] == "0"


def test_fem_info_flags_an_all_zero_load():
    """A form with no source solves perfectly to zero, residual and all. The load norm is what says so."""
    d = jno.shape.rect(0, 0, 1, 1, size=0.4).domain()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u, v = d.fem_symbols()
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    f = jno.fem([ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0])
    assert "ALL ZERO" in jno.info(f).as_dict()["operator"]["load ‖b‖"]


def test_solver_spec_info_names_what_it_does_and_drops_the_closures():
    data = jno.info(jno.solve.newton(direct=True)).as_dict()
    assert data["settings"]["direct"] == "True"
    assert not any("function" in str(v) for v in data["settings"].values()), "closures are not information"
    assert "theta" in jno.info(jno.solve.theta(0.5)).as_dict()["settings"]


def test_unhandled_object_says_what_is_handled():
    # a bare array IS handled now (it is a result), so this needs something genuinely foreign
    with pytest.raises(TypeError, match="Handled: a domain"):
        jno.info({"not": "a jno object"})


def test_the_retired_helpers_are_gone():
    d = _two_region_domain()
    assert not hasattr(d, "summary"), "domain.summary() was retired in favour of jno.info"
    assert not hasattr(jno.core, "print_tree")


# ---------------------------------------------------------------------------
# the environment report -- the two rows that silently ruin an answer
# ---------------------------------------------------------------------------


def test_env_info_reports_x64_device_and_versions():
    data = jno.info().as_dict()
    assert "float64 (x64)" in data["build"]
    assert data["build"]["float64 (x64)"] in ("ON",) or "OFF" in data["build"]["float64 (x64)"]
    assert "jax" in data["versions"]
    assert data["devices"], "at least one device must be reported"


def test_env_info_x64_row_carries_the_remedy_when_it_is_off():
    """Naming the flag is not enough -- it must be set before the first array, so the row says how."""
    import jax

    row = jno.info().as_dict()["build"]["float64 (x64)"]
    if not jax.config.jax_enable_x64:
        assert "jax_enable_x64" in row and "before the first array" in row


def test_fem_blocks_report_element_order():
    """A Taylor-Hood pair is P2/P1, and which block is which is the assembler's order, not yours."""
    d = jno.shape.rect(0, 0, 4, 1, size=0.5).domain()
    x, y, _ = d.variable("interior", split=True)
    l = d.variable("left", split=True)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p_, q_ = d.fem_symbols(names=("p", "q"))
    eu, ev = jno.np.symgrad(u, [x, y]), jno.np.symgrad(v, [x, y])
    pp, qq = p_.bind(x=x, y=y), q_.bind(x=x, y=y)
    f = jno.fem([jno.np.inner(eu, ev, n_contract=2) - pp * jno.np.trace(ev), -qq * jno.np.trace(eu),
                 u(l[0], l[1]) - (0.0, 0.0), p_.pin()])
    blocks = list(jno.info(f).as_dict()["field blocks"].values())
    assert any("P2" in b and "value_shape (2,)" in b for b in blocks)
    assert any("P1" in b for b in blocks)


# ---------------------------------------------------------------------------
# the small parts. These matter more than the assembled ones: the assembled object is where you
# find out something was wrong, and the small ones are where it went wrong.
# ---------------------------------------------------------------------------


def test_shape_info_before_meshing_reports_regions_and_the_csg_tree():
    sh = (jno.shape.rect(0, 0, 2, 1).name("lo").attach(k=5.0)
          + jno.shape.rect(0, 1, 2, 2).name("hi").attach(k=1.0)).sized(0.4)
    data = jno.info(sh).as_dict()
    assert "no" in data["geometry"]["meshed"], "an unmeshed shape must not claim mesh facts"
    assert data["regions"]["lo"].startswith("k=5.0")
    assert any("Rect" in v for v in data["CSG tree"].values())


def test_expression_info_names_the_region_it_samples():
    """A PDE residual bound to `boundary` instead of `interior` is invisible everywhere else."""
    d = jno.shape.rect(0, 0, 1, 1, size=0.5).domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u, v = d.fem_symbols()
    inside = jno.info((u.bind(x=x, y=y).x * v.bind(x=x, y=y).x)).as_dict()
    edge = jno.info((u.bind(x=xb, y=yb) * v.bind(x=xb, y=yb))).as_dict()
    assert inside["reads"]["regions"] == "interior"
    assert edge["reads"]["regions"] == "boundary"
    # both carry a trial AND a test, so both are weak-form terms
    assert "a jno.fem term" in inside["reads"]["weak form"]


def test_expression_info_counts_derivatives_and_the_tree():
    d = jno.shape.rect(0, 0, 1, 1, size=0.5).domain()
    x, y, _ = d.variable("interior", split=True)
    u, _v = d.fem_symbols()
    data = jno.info(u.bind(x=x, y=y).x).as_dict()
    assert int(data["structure"]["derivative nodes"]) >= 1
    assert "nodes" in data["structure"]["tree size"]


def test_variable_info_groups_the_components_of_one_tag():
    d = jno.shape.rect(0, 0, 4, 1, size=0.5).domain()
    rep = jno.info(d.variable("boundary"))
    data = rep.as_dict()
    spatial = [k for k in data if k.startswith("boundary")]
    assert spatial, "the spatial tag must appear once, not once per component"
    assert "component" in spatial[0]


def test_result_info_splits_by_the_forms_own_field_blocks():
    d = jno.shape.rect(0, 0, 4, 1, size=0.6).domain()
    x, y, _ = d.variable("interior", split=True)
    l = d.variable("left", split=True)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p_, q_ = d.fem_symbols(names=("p", "q"))
    eu, ev = jno.np.symgrad(u, [x, y]), jno.np.symgrad(v, [x, y])
    pp, qq = p_.bind(x=x, y=y), q_.bind(x=x, y=y)
    f = jno.fem([jno.np.inner(eu, ev, n_contract=2) - pp * jno.np.trace(ev), -qq * jno.np.trace(eu),
                 u(l[0], l[1]) - (0.0, 0.0), p_.pin()])
    sol = np.asarray(f.solve(linear=jno.solve.lu()))
    blocks = jno.info(sol, context=f).as_dict()["by field block"]
    assert len(blocks) == len(f.offsets) - 1
    assert sum(int(v.split()[0].replace(",", "")) for v in blocks.values()) == int(f.dofs)


def test_result_info_without_a_context_still_reports_the_array():
    data = jno.info(np.array([1.0, -2.0, 3.0])).as_dict()
    assert data["array"]["shape"] == "(3,)" and "-2" in data["array"]["range"]


# ---------------------------------------------------------------------------
# breadth: every mode, every route, and the degenerate arrays. A report that
# crashes on an empty array is a report nobody can rely on in a failure path --
# which is exactly when it is reached.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arr,expect", [
    (np.zeros(0), "EMPTY"),
    (np.zeros(5), "ALL ZERO"),
    (np.array([1.0, np.nan, np.inf, 2.0]), "non-finite"),
    (np.array(3.0), "()"),
])
def test_result_info_survives_degenerate_arrays(arr, expect):
    txt = str(jno.info(arr))
    assert expect in txt


def _rect(size=0.4, **kw):
    return jno.shape.rect(0, 0, 1, 1, size=size).domain(**kw)


def test_info_covers_every_fem_mode():
    d = _rect()
    x, y, _ = d.variable("interior", split=True)
    b = d.variable("boundary", split=True)
    u, v = d.fem_symbols()
    a, t = u.bind(x=x, y=y), v.bind(x=x, y=y)
    lin = jno.fem([a.x * t.x + a.y * t.y - 1.0 * t, u(b[0], b[1]) - 0.0])
    non = jno.fem([a.x * t.x + a.y * t.y + (a * a * a) * t - 1.0 * t, u(b[0], b[1]) - 0.0])
    assert jno.info(lin).as_dict()["form"]["mode"] == "linear"
    assert jno.info(non).as_dict()["form"]["mode"] == "nonlinear"


def test_complex_form_says_the_blocks_index_the_real_half():
    """dofs is 2n but offsets are n — slicing a solution on the block bounds would take half of it."""
    d = _rect()
    x, y, _ = d.variable("interior", split=True)
    b = d.variable("boundary", split=True)
    u, v = d.fem_symbols()
    a, t = u.bind(x=x, y=y), v.bind(x=x, y=y)
    f = jno.fem([a.x * t.x + a.y * t.y - (4.0 + 0.5j) * a * t, u(b[0], b[1]) - 0.0])
    row = jno.info(f).as_dict()["form"]["complex"]
    assert "REAL half" in row and "imaginary half follows" in row


def test_coupled_form_reports_one_block_per_field():
    d = _rect()
    x, y, _ = d.variable("interior", split=True)
    b = d.variable("boundary", split=True)
    T, s_ = d.fem_symbols(names=("T", "s"))
    C, r_ = d.fem_symbols(names=("C", "r"))
    Tb, sb = T.bind(x=x, y=y), s_.bind(x=x, y=y)
    Cb, rb = C.bind(x=x, y=y), r_.bind(x=x, y=y)
    f = jno.fem([Tb.x * sb.x + Tb.y * sb.y - Cb * sb, Cb.x * rb.x + Cb.y * rb.y - Tb * rb,
                 T(b[0], b[1]) - 0.0, C(b[0], b[1]) - 1.0])
    assert len(jno.info(f).as_dict()["field blocks"]) == 2


@pytest.mark.parametrize("structured", [True, False])
def test_fdm_info_names_the_route_it_took(structured):
    sh = jno.shape.rect(0, 0, 1, 1, size=0.25)
    d = (sh.structured() if structured else sh).domain()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=xi, y=yi)
    data = jno.info(jno.fdm([-ui.d2(xi) - ui.d2(yi) - 1.0, u(xb, yb) - 0.0])).as_dict()["solver"]
    assert ("structured" in data["route"]) is structured
    assert data["regime"] == "steady" and int(data["collocation points"]) > 0


@pytest.mark.parametrize("spec", [jno.solve.lu(), jno.solve.cg(), jno.solve.gmres(),
                                  jno.solve.bicgstab(), jno.solve.newton(), jno.solve.picard(),
                                  jno.solve.theta(0.5), jno.precond.jacobi()])
def test_every_solver_spec_reports_something(spec):
    assert len(str(jno.info(spec)).splitlines()) > 1


# ---------------------------------------------------------------------------
# jno.rcwa. This handler was written from attribute names and had NEVER been run;
# every one of its assumptions turned out wrong (see the commit). Hence a test.
# ---------------------------------------------------------------------------

_HAS_FMMAX = importlib.util.find_spec("fmmax") is not None


@pytest.mark.skipif(not _HAS_FMMAX, reason="fmmax (jno.rcwa backend) not installed")
def test_rcwa_info_before_and_after_solving():
    import jax.numpy as jnp

    from jno.trace.views import MatrixView

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        inner, vec = jno.np.inner, jno.np.vector
        K0, P, LZ, Z0, Z1 = 2 * jnp.pi, 0.6, 3.2, 0.8, 1.15
        d = jno.domain(jno.shape.box(0, 0, 0, P, P, LZ, size=0.3))
        u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
        c = d.variable("interior", split=True)
        xi, yi, zi = c[0], c[1], c[2]
        ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
        cu, cv = u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)
        nt, nb = d.variable("top", normals=True), d.variable("bottom", normals=True)
        cb = d.variable("bottom", split=True)
        tut, tvt = u.vector.cross(nt), v.vector.cross(nt)
        tub, tvb = u.vector.cross(nb), v.vector.cross(nb)
        einc = vec(1.0 + 0.0 * cb[0], 0.0 * cb[1], 0.0 * cb[2])

        def face(nm):
            cc = d.variable(nm, split=True)
            return u.bind(x=cc[0], y=cc[1], z=cc[2])

        e = jno.fn(lambda x, y, z: jnp.where((z >= Z0) & (z < Z1), 4.0, 1.0), [xi, yi, zi])
        eps = MatrixView(vec(e, e, e).expr).from_diag()
        cons = [inner(cu, cv) - K0**2 * inner(eps @ ui, vi),
                1j * K0 * inner(tut, tvt),
                1j * K0 * inner(tub, tvb) + 2j * K0 * inner(einc, tvb),
                face("left") - face("right"), face("front") - face("back")]

        rc = jno.rcwa(cons, orders=9)
        unsolved = jno.info(rc).as_dict()
        assert unsolved["setup"]["orders"] == "9"
        # the stack read back: vacuum ambient / eps-4 slab / vacuum ambient
        lay = list(unsolved["layers"].values())
        assert len(lay) == 3 and "semi-infinite" in lay[0] and "eps 4" in lay[1]
        assert "not solved" in str(unsolved["result"])

        solved = jno.info(rc.solve()).as_dict()["result"]
        # T + R = 1 for a lossless stack: the energy-conservation oracle, and the one number that
        # says whether the truncation order was enough.
        assert "energy conserved" in solved["T + R"]
        assert abs(float(solved["T + R"].split()[0]) - 1.0) < 1e-6
    finally:
        jax.config.update("jax_enable_x64", prev)


# ---------------------------------------------------------------------------
# A NEURAL OPERATOR (DeepONet) exercises four things a PINN does not: a replicated
# parametric domain, a parameter tag that is not a coordinate, a ModelCall rather
# than a bare Model, and an optimizer attached per-net instead of on the core.
# All four were wrong.
# ---------------------------------------------------------------------------


def _deeponet_problem(n_samples=4):
    import foundax
    import optax

    key = jax.random.PRNGKey(0)
    dom = n_samples * jno.shape.rect(0, 0, 2, 1, size=0.4).domain()
    x, y, _ = dom.variable("interior")
    k = dom.variable("k", jax.random.uniform(key, (n_samples, 1, 1), minval=0.5, maxval=1.5))
    net = jno.nn(foundax.deeponet(n_sensors=1, coord_dim=2, basis_functions=8, hidden_dim=32,
                                  activation=jax.numpy.tanh, key=key))
    net.optimizer(optax.adam(1e-3))
    u = net(k, jno.np.concat([x, y], axis=-1)) * x * (2 - x) * y * (1 - y)
    pde = k * (u.d2(x) + u.d2(y)) + 1.0
    return dom, k, net, pde, jno.core(constraints=[pde.mse])


def test_operator_residual_reports_the_network():
    """`net(...)` builds a ModelCall wrapping the Model -- an isinstance(Model) walk finds nothing,
    so a neural-operator residual reported no network at all."""
    _dom, _k, _net, pde, _crux = _deeponet_problem()
    assert "DeepONet" in jno.info(pde).as_dict()["reads"]["networks"]


def test_parameter_tag_is_not_reported_as_a_coordinate():
    """`dom.variable("k", values)` is a TensorTag with no mesh pool -- the coordinate report said
    nothing about it, and the generic expression report almost nothing."""
    _dom, k, _net, _pde, _crux = _deeponet_problem(n_samples=4)
    data = jno.info(k).as_dict()[""]
    assert "parameter" in data["kind"]
    assert data["samples"] == "4" and "shape (4, 1, 1)" in data["values"]


def test_core_finds_an_optimizer_attached_to_the_net():
    """`core.models` holds the UNWRAPPED module; `_opt_fn` lives on the Model wrapper. Looking in
    the wrong place told a user to call `.optimizer(...)` they had already called."""
    *_rest, crux = _deeponet_problem()
    assert "per-model" in jno.info(crux).as_dict()["training"]["optimizer"]


def test_replicated_domain_reports_its_sample_count():
    dom, *_rest = _deeponet_problem(n_samples=4)
    assert jno.info(dom).as_dict()["geometry"]["samples"] == "4"
