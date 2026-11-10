# Soft p-enrichment — where to raise the order, by gradient

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/soft_enrich_l_shape.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

`jno.solve.enrich()` ([adaptivity tutorial](adaptive-l-shape.md)) switches a node's interpolation cover
on or off with a boolean mask. A boolean carries no gradient, so *where* to enrich is invisible to
`jax.grad`. Here every node gets a weight $w_i \in [0, 1]$ instead, through a diagonal penalty on its
cover coefficients,

$$
A_w = A + \operatorname{diag}(p), \qquad p_k = A_{kk}\Bigl(\frac{1}{g_i} - 1\Bigr), \qquad g_i = w_{\min} + (1 - w_{\min})\,w_i ,
$$

so $g = 1$ is the full cover space, $g \to 0$ plain P1 at that node, and the Ritz energy
$\Pi(w) = \tfrac12 u(w)^\top A\, u(w) - b^\top u(w)$ — with $2(\Pi - \Pi_{\text{exact}})$ the squared
energy-norm error — is a smooth function of $w$ on **one** sparsity pattern, so one compiled program
serves every $w$.

At $w = 0$ its gradient is the diagonal hierarchical estimate (Bank & Smith, *SIAM J. Numer. Anal.* 30,
1993), $\partial\Pi/\partial w_i = -(1 - w_{\min}) \sum_{k \in i} r_k^2 / A_{kk}$, obtained by
automatic differentiation. The script checks that identity, then minimises $\Pi(w)$ under a budget
$\sum_i w_i = k$ by projected gradient and compares the thresholded optimum with the greedy `enrich`
loop **at the same enriched-node counts** (the budgets are that loop's rounds), and with uniform P1 at
the same active DOFs.

The problem is the one of the [adaptivity tutorial](adaptive-l-shape.md): $-\Delta u = f$ on an
L-shape, $u = 0$ on the boundary, a re-entrant corner and a smooth bump.

## What it shows

| enriched nodes $k$ | greedy `enrich` | budgeted optimum | uniform P1, same DOFs |
|---:|---:|---:|---:|
| 20 | 1.84e-3 | 1.62e-3 | 4.63e-3 |
| 38 | 1.07e-3 | 8.56e-4 | 4.03e-3 |
| 52 | 5.78e-4 | 4.97e-4 | 3.70e-3 |
| 63 | 3.80e-4 | 3.42e-4 | 3.46e-3 |
| 74 | 2.80e-4 | 2.47e-4 | 3.22e-3 |

Energy error $\Pi - \Pi_{\text{ref}}$ (lower is better), 203-node mesh. The optimum beats greedy marking
by 10–20 % at every budget, and from three random starts it lands on the same value to five digits. A
one-shot ranking by the $w = 0$ gradient is **worse** than greedy at every budget but the smallest: the gain comes from the coupling
between neighbouring covers, which the gradient at $w = 0$ cannot see. A SIMP gate $w^3$ is fine from
a uniform start but falls into local minima from random ones. The whole optimisation compiles a dozen
XLA programs; the greedy loop, which re-lowers whenever the mask changes, compiles a few hundred.

!!! note "Private for now"
    The weights live in `jno.utils.solver.fem_soft_cover`, which is not exposed through `jno.solve`;
    the script imports it directly. It covers one `space="cover"` field, steady and linear.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/soft_enrich_l_shape.py:code"
```
