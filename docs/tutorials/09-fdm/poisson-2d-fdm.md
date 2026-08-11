# Poisson 2D (FDM)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/09_fdm/poisson_2d_fdm.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

The strong-form counterpart of the [Poisson FEM primer](../08-fem-and-varpinns/poisson-2d-fem.md):
$-\Delta u = f$ on the unit square with $u = 0$ on the boundary, solved through `jno.fdm`. Instead of a
weak form, the strong residual is collocated at the mesh nodes with finite-difference stencils.

## Result

![Left: the jno.fdm solution u on the unit square (a smooth sine bump). Middle: the signed error u minus the analytic sin(pi x)sin(pi y). Right: a log-log mesh-refinement study of the relative L2 error versus mean element size.](/jNO/assets/poisson_2d_fdm.png)

The `jno.fdm` field matches the manufactured $u^\* = \sin(\pi x)\sin(\pi y)$ to
rel-$L^2 \approx 1.7\times10^{-2}$ on this mesh, and re-solving at four mesh sizes shows the error
falling at second order (fitted slope $\approx 2.07$, right) — the expected rate for the
finite-difference Laplacian.

## The constraint list

`u = domain.unknown()` is a valued nodal field — the strong-form counterpart of `fem_symbols()`.
Binding it gives the FD derivative views (`ui.d2(x)` is the finite-difference second derivative, no
`scheme=` needed), and the Dirichlet condition is the term `u(region) - g`:

```python
u  = d.unknown()
ui = u.bind(x=x, y=y)
f  = 2.0 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
sol = jno.fdm([
    -ui.d2(x) - ui.d2(y) - f,   # -Delta u = f
    u(xb, yb) - 0.0,            # Dirichlet u = 0
]).solve()
```

## What to notice

- One call, `jno.fdm([...])`, builds the strong-form system and `.solve()` returns the nodal field.
- The API mirrors `jno.fem`: `domain.unknown()` for the trial, `.bind` for derivatives, boundary
  conditions as terms in the same list.
- A nodal field's `.d` / `.d2` default to **finite differences** — autodiff is meaningless on a
  discrete field, so no `scheme=` is needed (pass one to pick a different stencil).
- The solution is audited against the manufactured field $u^\* = \sin(\pi x)\sin(\pi y)$
  (rel-$L^2 \approx 1.7\times10^{-2}$, the expected FD-discretization error on this mesh).

## Full script

```python
--8<-- "tutorial_examples/09_fdm/poisson_2d_fdm.py:code"
```
