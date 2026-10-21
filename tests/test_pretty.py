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


def test_numbers_are_recognised_as_fractions_of_known_constants_only_when_exact():
    """``3π²``, ``0.01/π``, ``√2/2`` and ``1/3`` as written in the formula, but only when they agree with the
    stored value to its precision: a number typed to four digits (``0.3183``) is not promoted to ``1/π``."""
    import math

    from jno.trace.pretty import _number, _Style

    u, a = _Style(False), _Style(True)
    float32 = 1e-6
    assert _number(3 * math.pi**2, u, float32) == "3π²"
    assert _number(0.01 / math.pi, u, float32) == "0.01/π"
    assert _number(math.sqrt(2) / 2, u, float32) == "√2/2"
    assert _number(1 / 3, u, 1e-12) == "1/3"
    assert _number(1 / (4 * math.pi), u, float32) == "1/(4π)"
    assert _number(0.3183, u, float32) == "0.3183"
    assert _number(0.9637, u, float32) == "0.9637"
    assert _number(3 * math.pi**2, a, float32) == "3*pi^2"


def test_deep_core_info_is_one_expression_tree_with_shapes():
    """``jno.info(crux, deep=True)``: every constraint as a tree in the same notation, sums flat, each node with
    its shape, nothing drawn twice, and the network defined once below the trees."""
    d, net, u, x, residual = _burgers()
    net.optimizer(optax.adam(1e-3))
    crux = jno.core([r.mse for r in residual], domain=d)
    lines = [v for _, v in jno.info(crux, deep=True).sections[-1][1]]
    assert jno.info(crux, deep=True).sections[-1][0] == "expression tree"
    n = np.shape(crux.domain.context["interior"])[-2]
    assert lines[0].startswith("[0] ∂u/∂t + u ∂u/∂x − 0.01 ∂²u/∂x²") and lines[0].endswith(f"({n}, 1) → mse → ()")
    body = [ln.strip() for ln in lines[1:]]
    assert any(b.startswith("├─ ∂u/∂t") for b in body)
    assert any(b.startswith("└─ −0.01 ∂²u/∂x²") for b in body)  # a subtracted term is a sibling with its sign
    # nothing is drawn twice: u is not a node of the tree, it is defined once, with its shapes
    assert not any(b.endswith("↑") or "MLP" in b for b in body[:-1])
    assert body[-1].startswith("where u = MLP(x, t)") and body[-1].endswith(f"({n}, 1)")


def test_solve_no_longer_logs_constraint_shapes(monkeypatch):
    """The shapes moved to ``jno.info(crux, deep=True)``; solve() used to log them, and trace every constraint
    an extra time to do it, on every call."""
    d, net, u, x, residual = _burgers()
    net.optimizer(optax.adam(1e-3))
    crux = jno.core([r.mse for r in residual], domain=d)
    said = []
    monkeypatch.setattr(crux.log, "info", lambda msg, *a, **k: said.append(str(msg)))
    crux.solve(2)
    assert not any("Shape =" in m for m in said), [m for m in said if "Shape" in m]
