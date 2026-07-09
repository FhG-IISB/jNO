# Differentiable Inverse (FDM)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/09_fdm/inverse_source_fdm.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/09-fdm/">Back to chapter</a>
</div>

When the constraint list carries a trainable `jno.np.parameter`, `jno.fdm([...]).solve()` returns a
differentiable **trace node** — exactly as `fem.solve()` does — so it composes straight into
`jno.core`. We recover the unknown amplitude $s$ of a source $s\,f_{\text{base}}$ from an observed
steady field, letting the parameter's own optimizer drive the fit.

## Result

![Left: the observed field u* (the synthetic data, a forward solve at true s=1). Middle: the fit residual at the recovered amplitude, at the 1e-8 level. Right: the recovered scalar's error |s-1| falling geometrically over gradient steps.](/jNO/assets/inverse_source_fdm.png)

The recovered quantity is a single scalar amplitude (the source $s\,f_{\text{base}}$ shares the known
basis $f_{\text{base}}$), so the recovery shows up as the fit, not a spatial field. From the wrong
start $s = 2.5$ the misfit gradient drives $s \to 1.0000$: the fit residual against the observation
collapses to $\sim\!10^{-8}$ (middle), and $|s-1|$ falls geometrically each SGD step until it saturates
at the iterative forward solver's tolerance floor ($\sim\!10^{-4}$, right).

## The solve is a node inside a `jno.core` loss

The parameter carries its optimizer; the solve goes straight into the misfit term — the same shape as
every FEM inverse tutorial:

```python
s = jno.np.parameter((1,), name="s")
s.optimizer(optax.adam(1e-1))
u = d.unknown(); ui = u.bind(x=x, y=y)

solve = jno.fdm([-ui.d2(x) - ui.d2(y) - s * f_base, u(xb, yb) - 0.0]).solve()   # a trace node
crux  = jno.core([(solve - observed).mse])          # domain inferred from the graph
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
--8<-- "tutorial_examples/09_fdm/inverse_source_fdm.py:code"
```
