# Miscellaneous

Features that extend jNO beyond the standard PINN workflow but don't belong under a single heading.

---

## Trackers

A **tracker** is an expression that is evaluated and logged each training step but **does not contribute to the loss**. Pass it to `jno.core` alongside the real constraints.

```python
from jno.numpy import tracker

val_error = tracker(jno.np.mean(jno.np.abs(u - u_exact)), interval=100)
crux = jno.core([pde.mse, bc.mse, val_error], domain)
```

`interval` controls how often the value is recorded (default: every step). The logged values appear in the `statistics` object returned by `solve()` alongside the loss curves.

Adaptive weight placeholders also expose `.tracker()`:

```python
w_pde, w_bc = jno.fn.adaptive.relobralo([pde.mse, bc.mse])
crux = jno.core([w_pde * pde.mse, w_bc * bc.mse, w_pde.tracker(), w_bc.tracker()], domain)
```

---

## Custom Functions

Wrap an arbitrary JAX function so it participates in the symbolic expression graph and can be differentiated, integrated, or included in losses:

```python
import jno.numpy as jnn

def my_fn(x, y):
    return jnp.exp(-x**2) * jnp.sin(y)

result = jnn.function(my_fn, [x, y])
```

`jnn.function` traces `my_fn` through JAX's tracing machinery, so gradients flow through it normally. Use this for nonlinear constitutive laws, lookup tables approximated by a JAX function, or any expression that is cleaner to write as a standalone function.

---

## Trainable Parameters

`jno.np.parameter` creates a **trainable scalar or small array** that participates in the optimisation loop exactly like a neural network — it gets an optimizer, and its value is updated each step.

```python
import jax, jno, optax

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)

a = jno.np.parameter((1,), key=k1, name="a")
b = jno.np.parameter((1,), key=k2, name="b")
c = jno.np.parameter((1,), key=k3, name="c")

for p in [a, b, c]:
    p.optimizer(optax.adam(1e-2))
```

The parameters can be used in any symbolic expression:

```python
residual = a * jno.np.sin(π * x) + b * jno.np.cos(π * x) + c * x * (1 - x) - target

crux    = jno.core([residual.mse], domain)
history = crux.solve(30000)

_a, _b, _c = crux.eval([a, b, c])
print(f"Recovered: a={_a[0]:.3f}, b={_b[0]:.3f}, c={_c[0]:.3f}")
```

This is the core building block for **inverse problems**: rather than fitting a field, the network identifies unknown PDE coefficients from residual constraints alone.

---

## Parameter Jacobian and Gradient Analysis

`u.grad(net)` computes the Jacobian of a traced expression with respect to the **trainable parameters** of a network — differentiating with respect to weights, not coordinates. Because it returns a `Placeholder`, it lives inside the expression graph and can be used anywhere a normal expression can: as a tracker logged every step, as a loss term, or evaluated after training.

| Symbol | Meaning |
|--------|---------|
| $B$ | batch (usually 1) |
| $N$ | collocation points |
| $P$ | trainable parameters (flattened) |

Shape is `(B, N, P)`, or `(B, N, D, P)` for multi-dimensional output.

### Tracking the Jacobian norm during training

The most common use is as a diagnostic tracker — log how the sensitivity of the solution to the network weights evolves across training:

```python
from jno.numpy import tracker

J    = u.grad(u_net)                              # (B, N, P)
J_norm = tracker(jno.np.mean(J ** 2), interval=50)  # scalar, logged every 50 steps

crux = jno.core([pde.mse, bc.mse, J_norm], domain)
history = crux.solve(10000)
```

### Tracking the Neural Tangent Kernel condition number

The NTK $K = J J^T$ governs convergence speed. Tracking its condition number during training reveals whether the problem is becoming better or worse conditioned:

```python
J      = u.grad(u_net)           # (B, N, P)
J_flat = J[0]                    # (N, P) — strip batch dim
K      = J_flat @ J_flat.T       # (N, N)

# Log the ratio of largest to smallest eigenvalue as a proxy for conditioning
K_norm = tracker(jno.np.max(K) / (jno.np.min(K) + 1e-8), interval=100)

crux = jno.core([pde.mse, K_norm], domain)
```

### Tracking gradient cosine similarity

Track whether the PDE residual and the boundary loss are pulling the weights in compatible directions throughout training:

```python
J_pde = pde_expr.grad(u_net)    # (B, N_int, P)
J_bc  = bc_expr.grad(u_net)     # (B, N_bc,  P)

g_pde = jno.np.mean(J_pde[0], axis=0)   # (P,) — average over interior points
g_bc  = jno.np.mean(J_bc[0],  axis=0)   # (P,)

cos_sim = tracker(
    jno.np.dot(g_pde, g_bc) / (jno.np.norm(g_pde) * jno.np.norm(g_bc) + 1e-8),
    interval=100,
)

crux = jno.core([pde.mse, bc.mse, cos_sim], domain)
```

A value near +1 means the two losses reinforce the same weight update each step; near −1 means they conflict and stall each other.

### Restricting to a parameter subset

For large networks the full Jacobian is expensive. Restrict to specific layers with a boolean mask:

```python
import equinox as eqx, jax

all_false   = jax.tree_util.tree_map(lambda _: False, u_net.module)
output_mask = eqx.tree_at(lambda m: m.output_layer.weight, all_false, True)

J_out  = u.grad(u_net.mask(output_mask))           # only output-layer weights
J_norm = tracker(jno.np.mean(J_out ** 2), interval=50)
```

### Using the Jacobian as a loss term

`u.grad(net)` can appear directly in a constraint, but differentiating *through* `jax.jacrev` is second-order AD and expensive. Use `.stop_gradient` to treat the current Jacobian as a constant regulariser:

```python
J_sg    = u.grad(u_net).stop_gradient
ntk_reg = (J_sg @ J_sg.T - target_K).mse
crux    = jno.core([pde.mse, ntk_reg], domain)
```
