# Flow Matching — eight-mode ring

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/13_generative/flow_matching_ring.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

Train a generative model by **conditional flow matching**, then sample from it by marching the trained network as an ODE. The whole training objective is one `.mse` term, and sampling is one call to `Model.flow`.

The target is a ring of eight Gaussian modes. It is chosen because it fails *loudly* in two independent ways: a collapsed velocity puts the sample radius somewhere other than 2, and a mode-dropping one leaves visible gaps around the circle.

Method: Lipman et al., ["Flow Matching for Generative Modeling"](https://arxiv.org/abs/2210.02747), ICLR 2023 — §3, the conditional objective.

## The idea

Flow matching learns a **velocity field** $v_\theta(x, t)$ whose flow carries a simple prior $p_0$ to the data $p_1$. Training never integrates anything: for a straight path between a prior draw $x_0$ and a data point $x_1$,

$$x_t = (1-t)\,x_0 + t\,x_1, \qquad \frac{dx_t}{dt} = x_1 - x_0,$$

the target velocity is *constant along the path*, so the loss is a plain regression of the network onto $x_1 - x_0$. Sampling is where the ODE appears.

## Step 1: Data and the probability path

```python
domain = jno.domain.from_array({"data": data})
d0, d1, _ = domain.variable("data")
x1 = jno.np.concatenate([d0, d1])          # the data sample

x0 = jno.noise.gaussian(std=1.0, like=x1)  # prior draw
t  = jno.noise.uniform(low=0.0, high=1.0, like=d0)   # path time, one per point

x_t    = (1 - t) * x0 + t * x1             # the linear (optimal-transport) path
target = x1 - x0                           # its velocity
```

`like=` takes the draw's shape from the referenced expression rather than a number written down. That matters here: the next step sub-samples the point axis, and a literal shape would go stale and fail by *broadcasting* rather than by raising — silently turning the residual into an outer product. `like=x1` tracks it automatically.

Note `x0` and `t` are two **independent** draws. That independence is the whole objective: if they were correlated, the loss would fall toward zero and the samples would diverge.

## Step 2: The velocity network

```python
v = jno.nn(foundax.mlp(D + 1, output_dim=D, hidden_dims=128,
                       num_layers=3, activation=jax.nn.swish, key=KEY))
v.optimizer(optax.adam(3e-3))
```

The network is called **state first, time second** — `v(x, t)` — which is the convention `Model.flow` marches with.

## Step 3: Train

```python
crux = jno.core([(v(x_t, t) - target).mse])
stats = crux.solve(epochs=4000, points=512)
```

That is the entire training objective. Note there is **no domain argument** — the loss collocates nothing, it is a fit over a dataset.

`points=512` mini-batches the **collocation-point axis**. A `from_array` dataset has exactly one sample, so `batchsize` has nothing to draw from here; `points=` is the axis that exists. See [`batchsize` and `points`](../../API.md#mini-batching-batchsize-and-points-are-different-axes).

## Step 4: Sample by marching the ODE

```python
z = jax.random.normal(jax.random.PRNGKey(1), (2000, D))
path = np.asarray(v.flow(z, time=(0.0, 1.0, 101), save=True))   # (101, 2000, D)
samples = path[-1]
```

`v.flow` integrates $dx/dt = v(x,t)$ with Heun's method. A **concrete** `x0` marches eagerly and returns the array, as here. A **symbolic** one returns a node that composes back into a `jno.core([...])` term list — which is what makes a PDE residual *on generated samples* expressible, and what lets a gradient reach an optimisable latent through the whole march.

`save=True` returns the trajectory rather than the endpoint; `time=` takes the grid points, so a decreasing grid marches backwards and maps data to its latent.

## Result

![Left: the eight-mode ring dataset. Middle: marched trajectories carrying prior draws to the data. Right: 2000 generated samples](/jNO/assets/flow_matching_ring.png)

Measured on this run:

| | |
|---|---|
| final CFM loss | **1.80** |
| generated radius | **2.024 ± 0.155** (data: 2.0) |
| mode occupancy | `[236 264 197 241 296 250 251 265]` — all eight populated |

**The loss is supposed to plateau around O(1).** Conditional flow matching has an *irreducible* loss: given $x_t$ and $t$ you cannot recover which $(x_0, x_1)$ pair produced them, so the regression target is genuinely stochastic. A loss near zero here is a **bug signal**, not convergence — it means the prior draw and the path time have stopped being independent, which looks like excellent training and produces samples that diverge to infinity.

## What To Notice

- **The objective is one term.** No sampler loop, no noise schedule object, no separate training script — the probability path is written as the formula it is, and `jno.core` compiles it.
- **`points=` is the right axis for a dataset.** Asking for `batchsize` larger than the sample count now raises and names `solve(points=N)`; it used to sample with replacement and evaluate duplicate copies of the same full point set at N times the cost, returning the full-batch loss with nothing said.
- **Training and sampling share one code path.** The eager march and the symbolic node both route through the same `march_velocity`, so `freeze` / `lora` / `dtype` apply identically and the two agree exactly, not approximately.
- **Accuracy is the step count.** Heun is fixed-step and second order — two network evaluations per step, error falling 4× per halving. There is no adaptive control and no error estimate.

## Limitations

Heun's method, fixed step, explicit. **No log-density** — that needs the divergence of $v$ with respect to the *state*, which `jno.np.divergence` (defined over coordinate `Variable`s) does not provide, so likelihood-based training and exact density evaluation are out of reach for now. Dropout must be off, since the march passes no PRNG key. Reverse mode stores every step, so a march *inside* a training loss costs its step count in both compute and activation memory — `core.solve(checkpoint_gradients=True)` trades roughly 30 % compute back for the memory.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/13_generative/flow_matching_ring.py" download>Download full script</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/13_generative/flow_matching_ring.py:code"
```
