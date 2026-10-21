"""Expressions in mathematical notation (`jno.trace.pretty`, shown by `jno.info`)."""

from __future__ import annotations

import foundax
import jax
import numpy as np
import optax

import jno
from jno.trace.pretty import equation, field_names, where_rows

KEY = jax.random.PRNGKey(0)


def _written(exprs, ascii_only=False):
    exprs = [getattr(e, "_expr", e) for e in exprs]
    names, defs = field_names(exprs)
    return [equation(e, names, ascii_only=ascii_only)[0] for e in exprs], [
        v for _, v in where_rows(defs, names, ascii_only=ascii_only)
    ]


def _burgers():
    d = jno.shape.rect(-1.0, 0.0, 1.0, 1.0, size=0.25).domain()
    x, t = d.variable("interior", split=True)[:2]
    net = jno.nn(foundax.mlp(2, 1, hidden_dims=8, num_layers=2, key=KEY))
    u = net(x, t.name("t"))
    return d, net, u, x, [u.d(t) + u * u.d(x) - 0.01 * u.d(x).d(x)]


def test_a_pinn_residual_reads_as_the_equation():
    """Coordinates by column (``x``) or by their ``.name`` (``t``), the network output as ``u``, derivative
    chains collapsed, a constant factor first."""
    *_, residual = _burgers()
    eqs, where = _written(residual)
    assert eqs == ["∂u/∂t + u ∂u/∂x − 0.01 ∂²u/∂x² = 0"]
    assert where == ["u = MLP(x, t)"]


def test_the_outputs_of_one_network_are_its_components():
    d = jno.shape.rect(0, 0, 1, 1, size=0.25).domain()
    x, y, _ = d.variable("interior")
    o = jno.nn(foundax.mlp(2, 3, hidden_dims=8, num_layers=2, key=KEY))(x, y)
    u, v, p = o[..., 0], o[..., 1], o[..., 2]
    eqs, where = _written([u * u.d(x) + v * u.d(y) + p.d(x) - u.laplacian(x, y) / 40, u.d(x) + v.d(y)])
    assert eqs == ["u₀ ∂u₀/∂x + u₁ ∂u₀/∂y + ∂u₂/∂x − Δu₀/40 = 0", "∂u₀/∂x + ∂u₁/∂y = 0"]
    assert where == ["(u₀, u₁, u₂) = MLP(x, y)"]


def test_a_differentiated_field_built_on_a_network_is_named_and_defined():
    """``net * x(1-x)`` is the field the equation is about: it takes the letter, the network keeps its class
    name inside the definition, and Laplacians of Laplacians collapse."""
    d = jno.shape.rect(0, 0, 1, 1, size=0.25).domain()
    x, y, _ = d.variable("interior")
    u = jno.nn(foundax.mlp(2, 1, hidden_dims=8, num_layers=2, key=KEY))(x, y) * x * (1 - x)
    eqs, where = _written([u.laplacian(x, y).laplacian(x, y) - 1.0])
    assert eqs == ["Δ²u − 1 = 0"]
    assert where == ["u = MLP(x, y) x (1 − x)"]


def test_signs_and_zeros_read_naturally():
    d = jno.shape.rect(0, 0, 1, 1, size=0.25).domain()
    x, y, _ = d.variable("interior")
    u = jno.nn(foundax.mlp(2, 1, hidden_dims=8, num_layers=2, key=KEY))(x, y)
    eqs, _ = _written([u - 0.0, u - (-2.0 * jno.np.sin(np.pi * x)), u + 0.0])
    assert eqs == ["u = 0", "u + 2 sin(π x) = 0", "u = 0"]


def test_ascii_when_the_terminal_cannot_show_the_symbols():
    *_, residual = _burgers()
    eqs, _ = _written(residual, ascii_only=True)
    assert eqs == ["du/dt + u du/dx - 0.01 d^2u/dx^2 = 0"]
    assert all(ord(ch) < 128 for ch in eqs[0])


def test_jno_info_of_a_core_lists_its_equations_with_region_and_points():
    d, net, u, x, residual = _burgers()
    net.optimizer(optax.adam(1e-3))
    rows = jno.info(jno.core([r.mse for r in residual], domain=d)).as_dict()["equations"]
    (key, value), *where = list(rows.items())
    assert key.startswith("[0] interior · ") and key.endswith(" pts")
    assert value == "∂u/∂t + u ∂u/∂x − 0.01 ∂²u/∂x² = 0   (mse)"
    assert where[0][1] == "u = MLP(x, t)"


def test_a_node_the_printer_does_not_know_prints_its_class_name():
    from jno.trace.pretty import render

    class Unknown(jno.trace.Placeholder):
        pass

    assert render(Unknown(), {}) == "Unknown(…)"
