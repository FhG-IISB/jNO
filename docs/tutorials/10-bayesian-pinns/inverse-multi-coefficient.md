# Inverse B-PINN: multi-coefficient regression

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/10_bayesian_pinns/02_inverse_multi_coefficient.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/10-bayesian-pinns/">Back to chapter</a>
</div>

**Pure inverse problem, no PDE residual.**  Recover two coefficients
`(A, B)` of a parametric model `d(x) = A sin(πx) + B cos(πx)` from
observations of `d`.  Each coefficient is configured with its own NUTS
kernel via `.bayesian(blackjax.nuts, ...)`; `solve()` dispatches them
in parallel.  Output: posterior mean and 90 % credible interval per
coefficient.

This is the cleanest demonstration of jNO's per-parameter Bayesian
configurator — every scalar carries its own kernel, no mixed mode, no
surrogate to worry about.  The same code shape works for any number of
coefficients.

## Script

```python
--8<-- "tutorial_examples/10_bayesian_pinns/02_inverse_multi_coefficient.py"
```
