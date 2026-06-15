# Fredholm Equation with Non-Separable Kernel

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/06_integration/fredholm_nonseparable.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/06-integration/">Back to chapter</a>
</div>

This example solves a Fredholm integral equation of the second kind whose kernel depends on **both** the evaluation point and the integration dummy simultaneously.  It requires the `.integrate(var=x)` API introduced for non-separable kernels.

## Equation

$$u(x) = f(x) + \int_0^1 (x + t)\, u(t)\, dt, \qquad x \in [0,1]$$

Exact solution: $u^*(x) = \sin(\pi x)$

### Derivation of f

$$\int_0^1 (x + t)\sin(\pi t)\, dt
  = x \underbrace{\int_0^1 \sin(\pi t)\, dt}_{2/\pi}
  + \underbrace{\int_0^1 t\sin(\pi t)\, dt}_{1/\pi}
  = \frac{2x}{\pi} + \frac{1}{\pi}$$

$$f(x) = \sin(\pi x) - \frac{2x}{\pi} - \frac{1}{\pi}$$

## Why this requires `.integrate(var=x)`

The kernel $K(x,t) = x + t$ is **non-separable in x**: for a fixed collocation point $x_i$, the integrand $(x_i + t)\,u(t)$ is a different function of $t$ for every $x_i$.  The result $\int_0^1 (x+t)\,u(t)\,dt$ is therefore an $(N,1)$ array — a function of $x$ — not a scalar.

With the separable trick (previous tutorial), you would split $K = x\cdot 1 + 1\cdot t$ and compute two independent scalar integrals.  With `.integrate(var=x)`, you write the kernel directly and jno handles the vectorisation via `jax.vmap`.

## Step 1: Two variables from the same domain call

```python
x, _ = domain.variable("interior")   # outer collocation variable
t, _ = domain.variable("interior")   # inner integration dummy  — no flag needed!
```

Both `x` and `t` point to the same mesh, but they are **distinct Python objects**.  The evaluator uses their object identity to decide which one to keep fixed (the one passed to `var=`) and which one to sweep (everything else).

## Step 2: The network appears at both roles

```python
u_x = net(x)   # evaluated at the N collocation points — what we want to learn
u_t = net(t)   # same weights, evaluated at the N integration points
```

The same trained weights power both evaluations.  Gradients flow through both `u_x` and the integral over `u_t` simultaneously.

## Step 3: Form the non-separable integral

```python
integral_term = ((x + t) * u_t).integrate(var=x)
```

`var=x` declares `x` as the outer variable.  The evaluator:

1. Fixes `x` at each collocation point via `jax.vmap`.
2. Evaluates $(x_i + t)\,u(t)$ over all mesh points for `t`.
3. Returns a weighted sum per outer point — shape `(N, 1)`.

The shape matches `u_x` and `f`, so the residual is formed naturally:

```python
residual = u_x - f - integral_term   # (N, 1)
```

## Step 4: Solve

```python
crux = jno.core([residual.mse])
crux.solve(30_000)
```

## Chaining two integrals (bonus)

Because `.integrate(var=x)` returns a standard `(N, 1)` placeholder, you can chain a second `.integrate()` on top to reduce it to a scalar.  For example, the iterated double integral

$$\int_0^1 \!\int_0^1 (x + t)\, dt\, dx = 1$$

can be verified in jno without any network:

```python
x, _ = domain.variable("interior")
t, _ = domain.variable("interior")

inner  = (x + t).integrate(var=x)   # (N, 1): g(x) = x + 0.5
result = inner.integrate()           # scalar: ∫₀¹ g(x) dx = 1.0
```

The inner call sweeps `t`; the outer scalar call then integrates `g(x)` over `x`.

## What to notice

- **No flag on `domain.variable()`.**  The only API change is `var=x` on `.integrate()`.
- **Object identity distinguishes roles.**  `x` and `t` are the same type with the same tag; what makes `x` the outer is that you pass it to `var=`.
- **N² network evaluations per step.**  For each of the N collocation points, the integrand is evaluated at all N integration points.  Keep the mesh coarse (or use a small MLP) to control cost.  `jax.vmap` + JIT compiles this to an efficient batched kernel.
- **Relative L2 error < 10 %** is achieved here with only 21 interior points and 30 000 steps.

## Script

```python
--8<-- "tutorial_examples/06_integration/fredholm_nonseparable.py"
```

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/06_integration/fredholm_nonseparable.py" download>Download full script</a>
<a class="md-button" href="/jNO/tutorials/06-integration/">Back to 06 Integration</a>
</div>
