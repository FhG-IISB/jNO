# Troubleshooting

Every message below was produced by running the code that provokes it, not copied from a source
file. jNO's rule is that a thing it cannot do **raises** rather than returning a plausible number, so
most of what you will hit is a refusal with the fix named in it — read to the end of the message
before changing anything.

## "no PDE residual found"

```python
jno.fdm([ui - 1.0, u(xb, yb) - 0.0])
# ValueError: jno.fdm([...]): no PDE residual found (a term with a derivative of the unknown).
```

A `jno.fdm` term list is the **strong** form: at least one term must carry a derivative of the
unknown. A list of pure algebraic conditions has no equation in it. The FEM analogue is a list with
no term carrying a test function.

## "Model 'MLP' … has no optimizer"

```python
crux = jno.core([(net(x) - 0.0).mse], domain=d)
crux.solve(1)
# ValueError: Model 'MLP' (layer 4) has no optimizer. Attach one with model.optimizer(...),
# sample its posterior with model.bayesian(...), fit a variational approximation with
# model.vi(...), or freeze the model with model.freeze().
```

Every trainable model needs one of those four declarations. jNO refuses to guess: a network with no
optimizer is as likely to be a frozen feature extractor as an oversight, and silently picking Adam
for it would train something you meant to hold fixed.

## "did not solve the system"

```
RuntimeError: jno.solve.fgmres did not solve the system: relative residual 2.8e-02 against a
1e-04 gate. This is the ADJOINT (transpose) solve, not the forward one — the solution itself is
fine and only the GRADIENT is affected. …
```

Every linear solve is residual-checked against its own operator, [including its
transpose](solvers.md#a-solve-that-did-not-converge-raises-including-the-adjoint). Read *which side*
failed: a forward failure means the problem needs a different solver or a preconditioner; an
**adjoint** failure means the solution is fine and only gradients are wrong, and a Krylov method that
breaks down on `Aᵀ` while converging on `A` is ordinary, not exotic.

!!! warning "Under `jit`, the message arrives wrapped"
    The check runs from inside the trace via a host callback, so a failure surfaces as
    `JaxRuntimeError: INTERNAL: CpuCallback error calling callback:` with the real `RuntimeError`
    **nested in the traceback below it**. Scroll past the XLA frames — the sentence naming the
    residual and the side is there.

## "the Krylov iteration broke down"

```
FloatingPointError: jno.solve.logdet: the Krylov iteration broke down and the result is not
finite (order=25, n=30). Lanczos/Arnoldi can only build a subspace as large as the number of
DISTINCT eigenvalues the probe sees …
```

A pinned FEM operator has far fewer distinct eigenvalues than rows — every Dirichlet DOF contributes
an identity row, so eigenvalue 1.0 carries the pinned count as its multiplicity. Lower `order` below
that count, or apply the estimator to the free-DOF operator. `applyfun` is immune (its `order` is an
upper bound); the stochastic three are not. See [Matrix functions](API.md#matrix-functions-what-ax-b-cannot-express).

## "Unknown differentiation scheme family"

```python
ui.laplacian(x, y, scheme="magic")
# ValueError: Unknown differentiation scheme family 'magic' (from scheme='magic'). Known:
# 'automatic_differentiation', 'finite_difference', 'spectral' …
```

The message lists every family and what each does. A sub-scheme goes after a colon —
`"finite_difference:cotangent"`, `"spectral:cosine"`.

## An optional backend is missing

```python
jno.solve.lu(backend="pardiso")(op, b)
# ImportError: jno.solve.lu(backend='pardiso') needs MKL PARDISO. Install it with
# `pip install jax-numerical-operators[fem]` …
```

Each optional backend raises by name with its install line. Run the
[verification snippet](Installation.md#check-what-you-actually-got) to see what you have before a
long job rather than after.

!!! note "A **dense** operator drops the backend, and says so in the log"
    `backend=` selects a *sparse* factorization. Hand `lu()` a dense operator and there is no sparse
    matrix to factor, so it falls back to `jnp.linalg.solve` — same answer, but not the backend you
    asked for, which matters if you chose it for speed. It logs that it did. An assembled `jno.fem`
    operator is always sparse, so this only affects a hand-built `LinearOperator(dense_array)`.

## A warning, not an error: the saddle-point default

```
UserWarning: jno.fem: p has no diagonal block, so this is a saddle-point system, and fem.solve()
is about to use its matrix-free default (Jacobi-preconditioned BiCGStab). A diagonal
preconditioner cannot help where the diagonal is zero: the solve may return a field whose
pressure has no correct digits while reporting success …
```

jNO detects a saddle system **structurally** at build time — a field whose test function never meets
its own trial function has no diagonal block. It warns rather than refuses, because a 2-D saddle of
moderate size does solve acceptably that way. Passing `linear=` or `precond=` silences it, since that
is the deliberate choice it is asking for. See [saddle preconditioning](solvers.md#preconditioners).

## Results are wrong, and nothing raised

The two things worth checking first, because neither can raise:

- **`jax_enable_x64` is off.** FEM assembly accumulates in float64 and an unstructured-mesh solve
  will not reach its tolerance without it. `jax.config.update("jax_enable_x64", True)` **before**
  building any domain, array or model — the flag affects only what is created after it.
- **The geometry is straight-sided.** By default a curved boundary is approximated to O(h²), which
  caps every element order above it. `Shape.curved()` fixes it. This and the axisymmetric-vector
  measure are the only two limits in `jno.fem` that are silent; every other one raises. See
  [Limits](fem/limitations.md).

## Where the message is not enough

`fem.stats` reports what the solver actually did — mode, DOFs, the slot reprs, the nonlinear driver's
final residual against its bound, and whether it converged. `fem.solve(profile=True)` says where the
time went. Both are on [Diagnostics](solvers.md#diagnostics-what-the-solver-actually-did).
