# Inverse Parameter

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/05_coupled_and_inverse/inverse_parameter.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/05-coupled-and-inverse/">Back to chapter</a>
</div>

This example is an inverse problem rather than a field solve: it learns unknown scalar coefficients from residual constraints.

## Problem Setup

The script introduces trainable scalar parameters and fits them so synthetic constraints are satisfied.

## Step 1: Treat Parameters as Learnable Objects

Instead of only training a neural field, the script creates scalar parameter models that participate in optimization.

```python
A_true, B_true, C_true = 3.14, -2.71, 42.0

domain = jno.domain.line(mesh_size=0.01)
x, _ = domain.variable("interior")

target = A_true * jno.np.sin(π * x) + B_true * jno.np.cos(π * x) + C_true * x * (1 - x)

k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
a = jno.np.parameter((1,), key=k1, name="a")
b = jno.np.parameter((1,), key=k2, name="b")
c = jno.np.parameter((1,), key=k3, name="c")
```

## Step 2: Build Residuals From Data Relationships

The optimization target is a set of algebraic or residual constraints rather than a spatial PDE field.

```python
residual = (a * jno.np.sin(π * x) + b * jno.np.cos(π * x) + c * x * (1 - x)) - target

for net in [a, b, c]:
    net.optimizer(optax.adam(1e-2))
```

## Step 3: Solve and Inspect Learned Coefficients

After optimization, the identified parameters are printed from the trained model set.

```python
crux    = jno.core([residual.mse])
history = crux.solve(30000)

_a, _b, _c = crux.eval([a, b, c])
print(f"Recovered parameters: a={_a[0]:.3f}, b={_b[0]:.3f}, c={_c[0]:.3f}")
```

## What To Notice

- jNO can optimize more than neural fields.
- Inverse problems often reuse the same core workflow with different residual definitions.
- This example is a good template for coefficient discovery and calibration.

## Going Further

For field identification (recovering a spatially-varying `k(x,y)` rather than a scalar), see the **[Inverse Problems](../../inverse-problems.md)** guide, which covers:

- `jno.fn.regularize` — smooth, TV, positivity and bounded penalties on identified fields
- `Model.constrain(transform)` — hard parameter constraints via paramax reparameterization
- `jno.domain.from_array` — attaching sparse sensor observations without file I/O

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/05_coupled_and_inverse/inverse_parameter.py" download>Download full script</a>
<a class="md-button" href="/jNO/tutorials/05-coupled-and-inverse/">Back to 05 Coupled and Inverse</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/05_coupled_and_inverse/inverse_parameter.py"
```
