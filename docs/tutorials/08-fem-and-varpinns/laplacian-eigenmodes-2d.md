# Dirichlet–Laplacian Eigenmodes (spectral verification)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/laplacian_eigenmodes_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

Most FEM checks drive an operator with a forcing term. This one verifies the assembled **mass** and
**stiffness** matrices *spectrally*, with no forcing at all: the generalized eigenproblem
$K\mathbf{x} = \lambda M\mathbf{x}$ on the interior degrees of freedom recovers the analytic Dirichlet
spectrum of $-\Delta$ on the unit square, $\lambda_{m,n} = \pi^2(m^2+n^2)$.

## Assembling the operators

Both matrices come straight from `jno.fem` — the mass from the `u*v` term, the stiffness from
`grad·grad` — and SciPy solves the generalized eigenproblem on the interior nodes:

```python
M = dense(jno.fem([ui * vi]).A)                          # mass: int phi_i phi_j
K = dense(jno.fem([ui.x * vi.x + ui.y * vi.y]).A)        # stiffness: int grad.grad
evals, V = sla.eigh(K[interior, interior], M[interior, interior])   # K x = lambda M x
```

![First four Dirichlet–Laplacian eigenmodes on the unit square.](/jNO/assets/laplacian_eigenmodes_2d.png)

## What to notice

- A spectral check catches mass/stiffness assembly errors a single forced solve cannot — a wrong
  matrix shifts the whole spectrum.
- The computed eigenvalues $\lambda/\pi^2 = 2.00, 5.03, 5.03, 8.06, 10.10$ match $2,5,5,8,10$, and the
  **degenerate** pair $5\pi^2$ (modes $(1,2)$ and $(2,1)$) is resolved as a near-equal pair.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/laplacian_eigenmodes_2d.py"
```
