# Residual norms: an H⁻¹ PINN loss

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/h_minus_1_residual_loss_poisson_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

The usual PINN loss is the mean square of the strong residual, `r.mse`. For a second-order operator such
as $-\Delta$, an error mode of frequency $k$ enters that loss with weight $\sim|k|^4$: gradient descent
clears the high frequencies at once and crawls along the smooth error, which is most of what the relative
error measures. Measuring the **same residual** in the dual norm $H^{-1}$,

$$
\|r\|_{H^{-1}}^2 = \langle r, (1 - \Delta)^{-1} r \rangle = \sum_k \frac{|\hat r_k|^2}{1 + \lambda_k},
$$

weights mode $k$ by $1/(1+\lambda_k)$, and the curvature grows like $|k|^2$ instead (SV-PINNs, Marcondes
2026, arXiv:2605.03542; the Deep Fourier Residual method, Taylor et al. 2022, arXiv:2210.14129).

## One changed line

```python
r = -u.laplacian(x, y) - f            # the strong residual

jno.core([r.mse])                     # the usual loss
jno.core([r.norm("H-1") ** 2])        # the same residual in H^-1
```

`norm(...) ** 2` is the squared norm (no square root in the gradient). The field is evaluated at every node
of its region, as `integrate` does.

## The spaces

| `space` | norm |
|---|---|
| `"l2"` | root mean square over the evaluated points; `r.norm("l2")**2` is `r.mse` |
| `"L2"`, `"L1"`, `"Linf"` | over the region with its measure (boundary tags: the boundary measure) |
| `"H1"`, `"H-1"`, `"H^s"` | $\langle r, (\text{scale} - \Delta)^s r\rangle^{1/2}$, `scale=1` by default (`scale=0`: the seminorm) |

`bc="dirichlet"` (default) takes test functions that vanish on the boundary (the dual of $H^1_0$);
`bc="neumann"` the dual of $H^1$. Without a space, `.norm()` is the pointwise norm, as before
(`|r|`, the Euclidean norm of a vector, `ord=` for a matrix). A vector or matrix field's Sobolev norm is
the sum over its components.

## The backend comes from the domain

- **Structured lattice** (`shape.structured(n=...)`): per-axis sine (Dirichlet) or cosine (Neumann)
  transforms, any real order `s` (`"H^-0.5"`). Dense transforms, O(n^(d+1)) per evaluation.
- **Simplex mesh** (1-D, 2-D, 3-D): P1 mass $M$ and stiffness $K$ built once;
  $\|r\|^2_{H^{-1}} = R^\top(M+K)^{-1}R$ with $R = M r$, one solve per evaluation: CG with an AMG
  preconditioner built once from the constant matrix (Jacobi, with a log line, when `pyamg` is not
  installed). `solver=` takes any `jno.solve` solver, `precond=` any `jno.precond` preconditioner. With AMG
  a training step costs 1.1–1.4× a plain `r.mse` step on a GPU (2-D, ~5k nodes); Jacobi up to 10×.
  Orders $s=\pm 1$ only.
- **Point cloud**, a shape domain without a mesh (`d.variable("interior", sample=(n, None))`): $H^{-1}$ as
  $\iint r(x)\,G(x-y)\,r(y)$ with $G$ the Green's function of $(\text{scale}-\Delta)$ in $\mathbb{R}^d$, by the point
  rule on the sampled points with weight $|\Omega|/N$ ($|\Omega|$ from the geometry); `L2`, `L1`, `Linf` the
  same way. No boundary condition (the whole-space norm of $r$ extended by zero); O(N²) per evaluation. No
  mesh is built (`method="mesh"` builds one).
- `method="grid"`, `"mesh"` or `"points"` forces one; the chosen backend is in `node.backend`.

Not yet: other orders than $H^{-1}$ on point clouds, the sampled SV-PINN estimator, periodic
`bc`, $H^s$ on boundary tags or sub-regions, gradients with respect to the mesh coordinates. Each raises an
error that says so.

Batched fields (`n * domain`, operator learning) and time-dependent fields are normed per batch sample and
per time level, like every constraint; the training loss averages them, as it does for `r.mse`.

## The result

Poisson $-\Delta u = f$ on the unit square with the SV-PINN paper's exact solution
$u = \sin\pi x \sin\pi y\,[\sin(a(x+y)) + \sin 2\pi x + \cos 3\pi y]$, $a = 10$, the same network (a 4 × 128 tanh MLP), Adam
schedule (×0.9 every 100 steps) and 2000 steps for both losses, on a 33 × 33 lattice (about 70 s on a CPU):

| loss | rel-$L^2$ error |
|---|---|
| `r.mse` | 9.7e-1 |
| `r.norm("H-1") ** 2` | 1.1e-1 |

One seed, one network size; a finer study (seeds, sizes, L-BFGS) is not part of this tutorial.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/h_minus_1_residual_loss_poisson_2d.py:code"
```
