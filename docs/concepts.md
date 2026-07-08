# Concepts

## One tracing system

jNO is built on a single idea: you **describe your problem as a symbolic expression**, not as a training
loop. Domain points, network calls, derivatives, PDE residuals, weak forms, integrals, noise, and
trainable parameters are all nodes in one **trace** — a symbolic graph. You hand that graph to
`jno.core(...)`, which JIT-compiles it **once** into a JAX function that is then reused for both
`crux.solve()` (training) and `crux.eval()` (evaluation). Because it is the same compiled graph, the very
same expression can serve as a residual *loss* during training and as a *quantity of interest* afterwards.

```python
u   = net(x)                       # a network call
pde = (u.dd(x) + f).mse            # a derivative + residual, reduced to a scalar
crux = jno.core([pde])             # compile the graph once
crux.solve(5000)                   # train through it
field = crux.eval([u])             # read the same graph back
```

## Why one graph covers PINN, NN, FEM, and FDM

The power of the trace is that four normally-separate workflows are just **different nodes in the same
graph**, so they compose freely and differentiate uniformly:

- **PINN** — the trial is a network and the loss is a strong-form PDE residual (`u.dd(x) + f`), with
  derivatives taken by automatic differentiation.
- **Plain NN / operator learning** — the loss is a supervised fit (`(pred - data).mse`); the same
  `jno.nn(...)` model, optimizer, and controls apply.
- **FEM** — the weak form is a list of residual terms handed to `jno.fem([...])`, which assembles the
  operator and exposes a **differentiable** `fem.solve()` node you can drop straight into the graph.
- **FDM** — the same derivative operators evaluate on the mesh via a finite-difference **scheme**
  (`u.d(x, scheme="finite_difference")`) instead of autodiff.

Because they are all nodes of one kind, you can mix them in a single `jno.core(...)` — a PINN residual, a
FEM solve, and a data term together — and **inverse problems fall out for free**: put a trainable
`jno.np.parameter` (or a whole network) anywhere in the graph, compare to data, and the gradient flows
back through the derivatives, the solve, and the network in one differentiable pass.

## The core vocabulary

| Term | What it is |
|------|------------|
| **Placeholder** | The base symbolic node — a coordinate, a network call, an operation, a residual. |
| **Constraint** | Any expression reduced to a scalar (`expr.mse`) and handed to `jno.core`. |
| **Crux** | The object `jno.core(...)` returns — holds the compiled step, optimizer state, and history. |
| **Model controls** | Per-model knobs (`optimizer`, `freeze`, `mask`, `lora`, `dtype`, …) on a wrapped `jno.nn(...)`. |

See the [Glossary](Glossary.md) for term-by-term definitions, and the [Getting Started](Getting-Started.md)
walkthrough for the smallest end-to-end version of the graph above. The design is described in
[`arXiv:2605.10159`](https://arxiv.org/abs/2605.10159).
