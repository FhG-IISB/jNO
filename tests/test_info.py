"""`jno.info` — one front door for what an object is.

It replaced three spellings on three classes (`domain.summary()`, `core.print_tree()`,
`core.print_shapes()`), so the first thing these tests pin is that the content those carried is
still reachable, and the second is that the cheap/deep split is real.
"""

from __future__ import annotations

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
    with pytest.raises(TypeError, match="Handled: a domain"):
        jno.info(np.zeros(3))


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
