"""``domain.unknown(space="spectral", order=...)`` and the points of a spectral problem, without a mesh.

A spectral unknown is a valued nodal field (``jno.fdm``'s kind of unknown) sized by its basis, not by a mesh,
so building one -- and every ``domain.variable`` of its problem -- must leave a lazy domain un-meshed. The
regions are resolved by tag; their point COUNTS are final at once, the positions are provisional until a
problem fixes the families from its ties.
"""

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


def _dom(time=None):
    return jno.shape.rect(0.0, -1.0, 2 * np.pi, 1.0).domain(**({"time": time} if time else {}))


def _meshed(d):
    return d.__dict__.get("_mesh") is not None


class TestUnknown:
    def test_no_mesh_is_built(self):
        d = _dom()
        U = d.unknown(space="spectral", order=(16, 9))
        x, y, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("bottom", split=True)
        assert not _meshed(d)
        assert U.model.module.value.shape == (16 * 9,)
        assert d.context[x.tag].shape == (1, 16 * 9, 2)
        assert d.context[xb.tag].shape == (1, 16, 2)

    def test_vector_unknown_and_basis(self):
        d = _dom()
        V = d.unknown.vector(2, space="spectral", order=8)
        assert V.model.module.value.shape == (64, 2)
        sp = V.basis
        assert sp.orders == (8, 8) and sp.box == ((0.0, 2 * np.pi), (-1.0, 1.0))
        assert not sp.resolved

    def test_explicit_families(self):
        d = _dom()
        U = d.unknown(space=("fourier", "legendre"), order=(8, 7))
        assert U.basis.explicit == ("fourier", "legendre")

    def test_derivatives_default_to_the_spectral_basis_family(self):
        d = _dom()
        U = d.unknown(space="spectral", order=(8, 7))
        x, y, _ = d.variable("interior", split=True)
        u = U.bind(x=x, y=y)
        assert str(u.x._expr.scheme).startswith("spectral_basis")
        assert str(u.xx._expr.scheme).startswith("spectral_basis")
        w = U.bind(x=x, y=y, scheme=jno.sp(dealias=1.5))
        assert w.x._expr.scheme.dealias == 1.5

    @pytest.mark.parametrize(
        "kw,match",
        [
            ({"complex": True}, "complex"),
            ({"constant": True}, "constant"),
            ({"order": (8,)}, "gives 1 sizes"),
            ({"order": 1}, "integer >= 2"),
            ({"space": ("fourier",)}, "names 1 axes"),
        ],
    )
    def test_refusals(self, kw, match):
        d = _dom()
        args = {"space": "spectral", "order": 8} | kw
        with pytest.raises((ValueError, NotImplementedError), match=match):
            d.unknown(**args)

    def test_non_box_domain_refuses(self):
        d = jno.shape.disk(0.0, 0.0, 1.0).domain()
        with pytest.raises(NotImplementedError, match="lives on a box"):
            d.unknown(space="spectral", order=8)

    def test_fem_unknowns_unchanged(self):
        d = jno.shape.rect(0, 0, 1, 1, size=0.25).structured().domain()
        u = d.unknown()
        assert u.model.module.value.shape[0] == d.mesh.points.shape[0]
        assert "_spectral_space" not in u.model.__dict__


class TestVariables:
    def test_regions_and_normals(self):
        d = _dom()
        d.unknown(space="spectral", order=(8, 5))
        xl, yl, _ = d.variable("left", split=True)
        xr, yr, _ = d.variable("right", split=True)
        P_l, P_r = d.context[xl.tag][0], d.context[xr.tag][0]
        assert np.allclose(P_l[:, 0], 0.0) and np.allclose(P_r[:, 0], 2 * np.pi)
        _xb, _yb, _t, nx, ny = d.variable("bottom", normals=True, split=True)
        assert np.allclose(d.context[nx.tag][0], [0.0, -1.0])
        assert d.__dict__["_spectral_tags"][xl.tag] == ("left", d._spectral_unknowns[0].basis)

    def test_nodes_of_another_unknown_get_their_own_tag(self):
        d = _dom()
        V = d.unknown.vector(2, space="spectral", order=(8, 9))
        P = d.unknown(space="spectral", order=(8, 7))
        xi, _, _ = d.variable("interior", split=True)
        xg, _, _ = d.variable("interior", nodes=P, split=True)
        xq, _, _ = d.variable("interior", nodes=jno.sp.gauss(y=7), split=True)
        assert xi.tag == "interior" and xg.tag != "interior" and xq.tag not in ("interior", xg.tag)
        assert d.context[xg.tag].shape == (1, 56, 2) and d.context[xq.tag].shape == (1, 56, 2)
        reg = d.__dict__["_spectral_tags"]
        assert reg[xg.tag] == ("interior", P.basis) and reg[xi.tag][1] is V.basis

    def test_initial_on_a_time_domain(self):
        d = _dom(time=(0.0, 1.0, 11))
        d.unknown(space="spectral", order=(8, 5))
        x0, y0, t0 = d.variable("initial", split=True)
        assert t0.tag == f"__time_{x0.tag}__"
        assert float(d.context[t0.tag][0, 0]) == 0.0
        assert not _meshed(d)

    def test_unknown_region_refuses_with_nodes(self):
        d = _dom()
        U = d.unknown(space="spectral", order=8)
        with pytest.raises(ValueError, match="a spectral region is one of"):
            d.variable("inlet", nodes=U)


def test_one_d_line_domain():
    d = jno.domain.line((0.0, 2.0), mesh_size=0.5)
    U = d.unknown(space="spectral", order=12)
    assert U.basis.box == ((0.0, 2.0),)
    xl, _ = d.variable("left")
    assert d.context[xl.tag].shape == (1, 1, 1)
