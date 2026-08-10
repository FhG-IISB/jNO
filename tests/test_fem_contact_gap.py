"""``u.gap(slave, master)`` — the signed contact gap symbol.

``g = g0 + n . (u_slave - u_master . Phi)`` at the slave face's quadrature points. It follows the
``domain.cell_size`` pattern: a placeholder ``Variable`` whose real per-quadrature-point value is packed
during assembly. What is covered here is the **symbol layer** — that a gap binds to two real boundary
regions, records the pairing for assembly, and refuses every way of getting it wrong.

The placeholder is deliberately *dropped* from the assembly context, so a gap that assembly has not
packed raises as an unresolved symbol rather than evaluating to zero — which would read as "everywhere
exactly in contact" and be believed.
"""

import jax
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    """x64: a penalty of 1e5 on the gap makes the system stiff enough that float32 cannot reach
    Newton's 1e-8 tolerance (measured: it stalls at a 5.7e-8 residual against a 1.3e-8 target)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _two_body_domain(size=0.4):
    """Two independently meshed blocks, so both sides of the interface are tagged separately."""
    return (
        jno.Shape.regions(
            lower=jno.Shape.box(0, 0, 0, 1, 1, 1),
            upper=jno.Shape.box(0, 0, 1, 1, 1, 2.5),
            conforming=False,
        )
        .sized(size)
        .domain()
    )


def _sides(d):
    return sorted(t for t in d.built_mesh.cell_sets if "|" in t)


def test_gap_binds_to_two_tagged_faces_and_records_the_pairing():
    d = _two_body_domain()
    u, _phi = d.fem_symbols(value_shape=(3,))
    slave, master = _sides(d)
    g = u.gap(slave, master, domain=d)

    assert f"gap_{slave}" in d.context, "the placeholder must exist so the Variable constructs"
    assert d._contact_pairs[f"gap_{slave}"][:2] == (slave, master), "assembly needs the pairing"
    assert getattr(g, "tag", None) == f"gap_{slave}"


def test_gap_requires_the_domain_explicitly():
    """A fem symbol carries no domain, and ``Placeholder`` turns attribute access into trace nodes —
    so a ``getattr(self, "_domain", None)`` fallback would return a *node*, not None, and quietly bind
    the Variable to nonsense. The argument is required and keyword-only to make that impossible."""
    d = _two_body_domain()
    u, _phi = d.fem_symbols(value_shape=(3,))
    slave, master = _sides(d)
    with pytest.raises(TypeError):
        u.gap(slave, master)  # positional/omitted -> keyword-only error
    with pytest.raises(TypeError, match="must be a jno domain"):
        u.gap(slave, master, domain="not a domain")


def test_gap_refuses_a_tag_that_is_not_a_boundary_region():
    d = _two_body_domain()
    u, _phi = d.fem_symbols(value_shape=(3,))
    slave, master = _sides(d)
    with pytest.raises(ValueError, match="not a boundary region"):
        u.gap("does_not_exist", master, domain=d)
    with pytest.raises(ValueError, match="not a boundary region"):
        u.gap(slave, "does_not_exist", domain=d)


def test_gap_refuses_a_face_against_itself():
    d = _two_body_domain()
    u, _phi = d.fem_symbols(value_shape=(3,))
    slave, _master = _sides(d)
    with pytest.raises(ValueError, match="must be different regions"):
        u.gap(slave, slave, domain=d)


def test_a_face_carries_at_most_one_gap():
    """Re-pairing the same slave face against a different master would silently overwrite the first
    pairing, so the second contact would quietly use the wrong master surface."""
    d = _two_body_domain()
    u, _phi = d.fem_symbols(value_shape=(3,))
    slave, master = _sides(d)
    u.gap(slave, master, domain=d)
    u.gap(slave, master, domain=d)  # idempotent: the same pairing again is fine
    with pytest.raises(ValueError, match="already the slave face"):
        u.gap(slave, "top", domain=d)


def test_a_normal_gap_needs_a_vector_field():
    """`n . (u_s - u_m)` is a vector contraction. With a SCALAR field `jnp.einsum("d,qd->q", n, jump)`
    silently broadcasts the size-1 component axis and computes `sum(n) * jump` — right only when the
    normal happens to sum to 1, wrong on any tilted interface, and never an error. Refuse instead."""
    d = _two_body_domain()
    slave, master = _sides(d)
    scalar, _ = d.fem_symbols()
    with pytest.raises(ValueError, match="needs a vector field"):
        scalar.gap(slave, master, domain=d)
    vec, _ = d.fem_symbols(value_shape=(3,))
    assert vec.gap(slave, master, domain=d) is not None


def _vector_poisson(c=None, size=0.4):
    """Vector Poisson on the two-body bar; ``c`` adds a penalty on the interface gap. Returns the
    largest jump in u_z across coincident interface node pairs, and the peak displacement."""
    d = _two_body_domain(size)
    u, v = d.fem_symbols(value_shape=(3,))
    slave, master = _sides(d)
    ci = d.variable("interior", split=True)
    bb = d.variable("boundary", split=True)
    vi = v.bind(x=ci[0], y=ci[1], z=ci[2])
    gu = jno.np.grad(u, [ci[0], ci[1], ci[2]])
    gv = jno.np.grad(v, [ci[0], ci[1], ci[2]])
    terms = [jno.np.inner(gu, gv, n_contract=2) - 1.0 * vi[2]]
    if c is not None:
        sb = d.variable(slave, split=True)
        n = d.variable(slave, normals=True)
        g = u.gap(slave, master, domain=d)
        terms.append(c * g * jno.np.inner(n, v.bind(x=sb[0], y=sb[1], z=sb[2]), n_contract=1))
    terms += [u(bb[0], bb[1], bb[2])[i] - 0.0 for i in range(3)]

    sol = np.asarray(jno.fem(terms, element_type="TET4").solve()).reshape(-1, 3)
    pts = np.asarray(d.built_mesh.points)
    lo = np.asarray(d.tag_indices[slave]).reshape(-1)
    up = np.asarray(d.tag_indices[master]).reshape(-1)
    key = {tuple(np.round(pts[i, :2], 9)): i for i in lo}
    pairs = [(key[k], j) for j in up if (k := tuple(np.round(pts[j, :2], 9))) in key]
    assert len(pairs) > 5
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    return float(np.abs(sol[a, 2] - sol[b, 2]).max()), float(np.abs(sol).max())


def test_a_penalty_on_the_gap_closes_the_interface_like_one_over_c():
    """The end-to-end gate, and the one that proves the gap is *live*: penalising it must drive the
    interface jump to zero like 1/c. That can only happen if the gap measures the real jump AND the
    tangent carries the master-side coupling — the matrix-free path picks the latter up through
    `jax.linearize` of the residual, which is why no assembled Jacobian block is built."""
    free, _ = _vector_poisson(None)
    jumps = {c: _vector_poisson(c)[0] for c in (1e2, 1e3, 1e4, 1e5)}

    assert jumps[1e2] < 0.5 * free, "a penalty must reduce the jump at all"
    for lo, hi in ((1e2, 1e3), (1e3, 1e4), (1e4, 1e5)):
        ratio = jumps[lo] / jumps[hi]
        assert 3.0 < ratio < 30.0, f"expected ~10x per decade of c, got {ratio:.1f} ({lo}->{hi})"
    assert jumps[1e5] < 1e-3 * free


def test_the_assembled_tangent_refuses_a_gap():
    """A gap is non-local, and the per-element Jacobian emits parent-cell columns only, so it would
    silently drop the slave–master coupling. It must refuse rather than return a degraded tangent."""
    d = _two_body_domain()
    u, v = d.fem_symbols(value_shape=(3,))
    slave, master = _sides(d)
    ci = d.variable("interior", split=True)
    sb = d.variable(slave, split=True)
    n = d.variable(slave, normals=True)
    gu = jno.np.grad(u, [ci[0], ci[1], ci[2]])
    gv = jno.np.grad(v, [ci[0], ci[1], ci[2]])
    g = u.gap(slave, master, domain=d)
    terms = [
        jno.np.inner(gu, gv, n_contract=2) - 1.0 * v.bind(x=ci[0], y=ci[1], z=ci[2])[2],
        g * jno.np.inner(n, v.bind(x=sb[0], y=sb[1], z=sb[2]), n_contract=1),
    ]
    bb = d.variable("boundary", split=True)
    terms += [u(bb[0], bb[1], bb[2])[i] - 0.0 for i in range(3)]
    fem = jno.fem(terms, element_type="TET4")
    with pytest.raises(NotImplementedError, match="matrix-free tangent"):
        fem.jacobian(np.zeros(fem.n_dofs) if hasattr(fem, "n_dofs") else np.zeros(1))
