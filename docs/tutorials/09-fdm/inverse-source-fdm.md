# Differentiable Inverse (FDM)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/09_fdm/inverse_source_fdm.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/09-fdm/">Back to chapter</a>
</div>

When the constraint list carries a trainable `jno.np.parameter`, `jno.fdm([...]).solve()` returns a
differentiable **trace node** — exactly as `fem.solve()` does — so it composes straight into
`jno.core`. We recover the unknown amplitude $s$ of a source $s\,f_{\text{base}}$ from an observed
steady field, letting the parameter's own optimizer drive the fit.

## The solve is a node inside a `jno.core` loss

The parameter carries its optimizer; the solve goes straight into the misfit term — the same shape as
every FEM inverse tutorial:

```python
s = jno.np.parameter((1,), name="s")
s.optimizer(optax.adam(1e-1))
u = d.unknown(); ui = u.bind(x=x, y=y)

solve = jno.fdm([-ui.d2(x) - ui.d2(y) - s * f_base, u(xb, yb) - 0.0]).solve()   # a trace node
crux  = jno.core([(solve - observed).mse], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
crux.solve(150)
```

## What to notice

- No adjoint code and no manual gradient loop: `crux` drives the parameter, and the gradient flows
  through the solve's implicit `custom_root` — the same mechanism `fem.solve()` uses.
- `jno.fdm([...]).solve()` is still the one entry: **with** a trainable parameter it is a trace node
  the crux re-runs each step; **without** one it returns the solution array eagerly.
- It is a **twin experiment** — the observation is the forward solve at the true `s = 1`, so the
  minimizer is exactly the truth: from a wrong start $s = 2.5$ the fit recovers $s \approx 0.999$.

## Full script

```python
--8<-- "tutorial_examples/09_fdm/inverse_source_fdm.py"
```
