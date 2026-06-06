# API Reference

This page is auto-generated from in-source docstrings via
[mkdocstrings](https://mkdocstrings.github.io/). When something here
looks wrong, fix the docstring — not this file.

For conceptual prose (what these objects *are* and *why* they exist),
see the [Getting Started](Getting-Started.md) guide and the
[Concepts](Glossary.md) page.

---

## `jno.core`

The top-level solver. Wraps a list of constraint expressions and a
domain, compiles them once, then exposes `solve()` / `eval()`.

::: jno.core.core
    options:
      members:
        - __init__
        - compile
        - solve
        - eval
        - sweep
        - print_shapes

---

## Domain

`jno.domain` is the entry point for spatial geometry, mesh management,
sampling, and tensor tags.

::: jno.domain.domain
    options:
      members:
        - __init__
        - variable
        - sample
        - add_tensor_tag
        - summary
        - line
        - rect
        - polygon
        - export
        - plot_mesh

::: jno.PolygonDomain

---

## Neural-network controls

`jno.nn.wrap` lifts a plain Equinox / foundax module into a jNO
`Model` so it can participate in the trace and accept per-model
optimisers, masks, LoRA, freezing, and so on.

::: jno.architectures.models.nn
    options:
      members:
        - wrap

::: jno.trace.Model
    options:
      members:
        - optimizer
        - lr
        - freeze
        - unfreeze
        - mask
        - lora
        - dtype
        - constrain
        - initialize
        - tune

---

## Symbolic math (`jno.np`)

A NumPy-compatible namespace that returns traced placeholders instead
of concrete arrays. Use it inside any expression that you intend to
feed into `jno.core(...)`.

::: jno.jnp_ops
    options:
      members:
        - concat
        - sin
        - cos
        - exp
        - log
        - sqrt
        - abs
        - grad
        - pi

---

## Function helpers and loss balancers (`jno.fn`)

`jno.fn` provides PDE-named helpers (`heat`, `wave`, `burgers_1d`, ...),
loss reductions (`mse`, `mae`, `rmse`, `huber`, `log_cosh`, ...), and
the adaptive loss balancers under `jno.fn.adaptive.*`.

::: jno.fn
    options:
      heading_level: 3
      members:
        - _module_call

---

## Training history

`solve()` returns a `statistics` object. The most common operations:

::: jno.utils.statistics.statistics
    options:
      members:
        - total_loss
        - total_loss_history
        - plot
        - save
        - load

---

## Differential and integral operators

These provide the residuals you put inside constraints
(`u.laplacian(x, y)`, `u.d(x)`, `(grad_u * n).integrate()`).

::: jno.differential_operators.DifferentialOperators

::: jno.integration_operators.IntegrationOperators

---

## Tracing primitives

Most users never instantiate these directly — they are what the
expression-building API returns. Documented here for reference and for
authors of custom operators.

::: jno.trace.Variable

::: jno.trace.Integral

::: jno.trace.Noise
