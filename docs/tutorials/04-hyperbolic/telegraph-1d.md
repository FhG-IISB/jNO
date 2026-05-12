# Telegraph 1D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/telegraph_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to chapter</a>
</div>

This example adds damping to a wave equation and produces the classical telegraph equation.

## Problem Setup

The PDE has the form `u_tt + beta u_t = c^2 u_xx + f`, with manufactured data used for validation.

## Step 1: Reuse the Wave-Equation Structure

The script keeps the second-order time derivative but introduces an additional first-order time term.

## Step 2: Handle Two Initial Conditions

As in the wave example, both displacement and velocity information at `t = 0` are required.

## Step 3: Add Damping to the Residual

The `beta u_t` term changes the dynamics from undamped propagation to dissipative wave motion.

## What To Notice

- This is a minimal extension of the wave equation that changes the qualitative behavior.
- Damping is straightforward to add once time derivatives are already available.
- The example is useful for understanding how multiple time-derivative orders coexist in one residual.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/telegraph_1d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to 04 Hyperbolic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/04_hyperbolic/telegraph_1d.py"
```
