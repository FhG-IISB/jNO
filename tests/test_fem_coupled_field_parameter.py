"""A nodal FEM field parameter k(x) on a coupled (multi-field) problem (Phase 7 of the compose work).

Previously ``jno.fem (native)`` rejected a ``jno.np.parameter(phi)`` coefficient *field* on any
multi-field problem (single-field only). It now associates the field parameter with the field whose
test function appears in the term(s) referencing it (e.g. ``k(x)·(grad p . grad q)`` -> the ``p`` field),
so its nodal values gather/interpolate on **that** field's FE space. A field parameter *shared* across
several fields' terms is allowed when those fields share ONE FE space (same element order -> identical
nodes, so k interpolates the same on any of them) — a material property common to coupled equations;
only fields of DIFFERING order (no shared node set) are rejected. (Coupled-steady-parametric is blocked
upstream regardless, so this exercises the coupled *transient* native path, which threads runtime
parameters per step.)

Correctness mirrors the single-field nodal-parameter test: a *linear* nodal field must assemble the
same spatial operator as the equivalent coordinate-function coefficient (P1 interpolation of a linear
field is exact), which catches any gather/node-order/field-index error.

Run with x64 (FEM assembly is float64).
"""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def _build(use_param, span_both=False):
    """Coupled transient diffusion of ``u`` (field 0) and ``p`` (field 1); ``p``'s diffusion is scaled by
    ``k`` -- a nodal field parameter on ``p``'s FE space (``use_param``) or the equivalent coordinate
    function ``0.6 + 0.8x + 0.5y``. ``span_both`` also scales ``u``'s diffusion by ``k`` (a parameter
    referenced by two fields -> the ambiguous case)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.15, time=(0.0, 0.02, 6))
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    pi, qi = p.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
    k = jno.np.parameter(q, name="k") if use_param else (0.6 + 0.8 * xi + 0.5 * yi)
    u_diff = (k if span_both else 1.0) * (ui.x * vi.x + ui.y * vi.y)
    cons = [
        ui.t * vi + u_diff,
        pi.t * qi + k * (pi.x * qi.x + pi.y * qi.y),  # k on the p (field-1) equation
        u(xb, yb) - 0.0,
        p(xb, yb) - 0.0,
        u(ci[0], ci[1]) - 0.0,
        p(ci[0], ci[1]) - 0.0,
    ]
    return d, jno.fem(cons)


def test_coupled_field_parameter_on_second_field_matches_coordinate_coeff():
    """A nodal field parameter k(x) on the *second* field assembles (previously rejected) and produces
    the same coupled spatial operator as the equivalent coordinate-function coefficient -- proving the
    field parameter is gathered/interpolated on the right field's FE space (P1 interpolation of a linear
    field is exact). Compared on the non-Dirichlet rows: a Dirichlet row is a pinned identity whose
    representation differs between the parametric ``operator_fn`` and the constant ``.A`` (it differs in
    the param-free u-block too), so it carries no field-parameter information."""
    d, fem = _build(use_param=True)
    assert fem.is_transient
    assert list(fem.operator.runtime_parameter_exprs) == ["k"], "k must be a (single) runtime parameter"

    # k(x) on field 1's (p) P1 nodes -> the linear field, exact under P1 interpolation
    pts1 = np.asarray(fem.field_points[1])
    k_true = jnp.asarray(0.6 + 0.8 * pts1[:, 0] + 0.5 * pts1[:, 1])
    A_param = _dense(fem.operator.operator_fn(0.0, {"k": k_true}))

    _, fem_ref = _build(use_param=False)  # coordinate-function coefficient (same mesh_size -> same mesh)
    A_ref = _dense(fem_ref.operator.A)

    # free (non-Dirichlet) DOFs across both field blocks: where the assembled physics actually lives.
    # A Dirichlet DOF is a pinned identity row+column whose representation differs between operator_fn
    # and .A (in the param-free u-block too), so restrict the comparison to the free×free submatrix.
    n = int(fem.offsets[1])
    bnd = set(int(i) for i in np.asarray(d.tag_indices["boundary"]).ravel())
    interior = np.array([i for i in range(n) if i not in bnd], dtype=int)
    free = np.concatenate([interior, n + interior])
    diff = np.max(np.abs(A_param[np.ix_(free, free)] - A_ref[np.ix_(free, free)]))
    assert diff < 1e-9, f"field-parameter operator != coordinate-coeff operator on free DOFs: {diff:.2e}"


def test_field_parameter_spanning_two_same_order_fields_matches_coordinate_coeff():
    """A field parameter *shared* by two same-order (P1) fields' terms now assembles (a material property
    common to coupled equations) and produces the same coupled operator as the equivalent coordinate
    coefficient — the shared k(x) interpolates identically on either field's (identical) P1 space."""
    d, fem = _build(use_param=True, span_both=True)
    assert fem.is_transient
    assert list(fem.operator.runtime_parameter_exprs) == ["k"], "k must be a (single) runtime parameter"

    # k now interpolates on the shared P1 space (min referencing field idx); both fields' P1 nodes coincide
    pts0 = np.asarray(fem.field_points[0])
    k_true = jnp.asarray(0.6 + 0.8 * pts0[:, 0] + 0.5 * pts0[:, 1])
    A_param = _dense(fem.operator.operator_fn(0.0, {"k": k_true}))

    _, fem_ref = _build(use_param=False, span_both=True)  # coordinate coeff in BOTH equations
    A_ref = _dense(fem_ref.operator.A)

    n = int(fem.offsets[1])
    bnd = set(int(i) for i in np.asarray(d.tag_indices["boundary"]).ravel())
    interior = np.array([i for i in range(n) if i not in bnd], dtype=int)
    free = np.concatenate([interior, n + interior])  # free DOFs across BOTH field blocks (k scales both)
    diff = np.max(np.abs(A_param[np.ix_(free, free)] - A_ref[np.ix_(free, free)]))
    assert diff < 1e-9, f"shared field-parameter operator != coordinate-coeff operator on free DOFs: {diff:.2e}"


def test_field_parameter_spanning_different_order_fields_raises():
    """A field parameter shared by fields of DIFFERENT order (P1 + P2) has no common node set to
    interpolate on -> a clear NotImplementedError (rather than a silently-wrong gather)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2, time=(0.0, 0.02, 4))
    u, v = d.fem_symbols(names=("u", "v"))  # P1
    p, q = d.fem_symbols(names=("p", "q"), order=2)  # P2
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    pi, qi = p.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
    k = jno.np.parameter(v, name="k")  # a P1 field parameter (field params are P1-only) scaling BOTH the
    #                                    P1 (u) and P2 (p) diffusion -> spans differing orders
    cons = [
        ui.t * vi + k * (ui.x * vi.x + ui.y * vi.y),
        pi.t * qi + k * (pi.x * qi.x + pi.y * qi.y),
        u(xb, yb) - 0.0,
        p(xb, yb) - 0.0,
        u(ci[0], ci[1]) - 0.0,
        p(ci[0], ci[1]) - 0.0,
    ]
    with pytest.raises(NotImplementedError, match="same element order|differing orders|one FE space"):
        jno.fem(cons)
