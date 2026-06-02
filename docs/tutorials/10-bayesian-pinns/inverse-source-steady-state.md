# Inverse B-PINN: source recovery in an elliptic PDE

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/10_bayesian_pinns/03_inverse_source_steady_state.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/10-bayesian-pinns/">Back to chapter</a>
</div>

**Mixed mode.**  Recover an unknown source amplitude `A` in the
steady-state heat equation `α u''(x) + A sin(πx) = 0` from a synthetic
observation of `u`.  The amplitude is sampled with NUTS while an MLP
surrogate `u(x)` is trained with Adam.  In mixed-mode setups the
window-adaptation default is bypassed (`adapt=False`) because adapter
runs against an untrained surrogate produce mis-tuned hyperparameters —
see [the mixed-mode caveat](../../training/bayesian.md#mixed-mode).

The transient version (`u_t = α u_xx + A sin(πx)`) follows the same
recipe with `jno.domain.line(..., time=(0, T_end, N))` and an ansatz
that hard-enforces the initial condition, at the cost of a noticeably
longer JIT compile.

## Script

```python
--8<-- "tutorial_examples/10_bayesian_pinns/03_inverse_source_steady_state.py"
```
