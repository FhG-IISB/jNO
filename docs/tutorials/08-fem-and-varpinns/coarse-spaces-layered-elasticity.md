# Coarse Spaces for Schwarz (layered elasticity)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/coarse_spaces_layered_elasticity.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

Overlapping Schwarz solves the problem on many small subdomains at once. Information travels one
subdomain per iteration, so the CG iteration count grows with the number of subdomains. A **coarse space**
fixes this. It names the few slowly varying modes that the local solves cannot reach, and solves for them
globally at every iteration. This tutorial builds the coarse space for elasticity from its rigid-body
modes. It then shows the one case they do not cover, a stiffness jump that cuts through the subdomains, and
the coarse space that does: GenEO.

The problem is plane-strain elasticity on a $4\times1$ block, clamped on the left and loaded by its own
weight. It is made of four layers whose stiffness alternates between $E=10^4$ and $E=10^4/c$. A contrast
$c=1$ is a homogeneous block.

## One value for the rigid-body modes

The coarse space is a **value**. Here it is the three rigid-body modes, written as formulas in the domain's
coordinates. `parts=S` cuts them to the subdomains of the Schwarz preconditioner `S`, which gives three
vectors per subdomain. `Q >> S >> Q` is the balancing Neumann–Neumann combination: coarse solve, local
solves, coarse solve. It is symmetric, so plain CG applies:

```python
rigid = [(1, 0), (0, 1), (-y, x)]                     # two translations, one rotation
S = jno.precond.schwarz(parts=64, coarse=False)       # one-level: local solves only
Q = jno.precond.coarse(rigid, parts=S)                # the modes on S's 64 subdomains
fem.solve(linear=jno.solve.cg(tol=1e-8), precond=Q >> S >> Q)
```

The same `rigid` works for any number of subdomains, any mesh and either material, because it is
evaluated on the problem when the solve sets up. Rows eliminated by the clamp get zeros, and a vector
that vanishes entirely is dropped and logged.

## What to notice

- **One level grows; two levels don't.** For the homogeneous block, one-level Schwarz takes 55, 110
  and 170 iterations at 4, 16 and 64 subdomains. With the rigid-body modes it takes 46, 56 and 45.
- **The contrast leaves a gap that rigid modes cannot close.** At $c=10^4$ the rigid modes still stop
  the growth (73, 136 and 110, against 80, 195 and 292 for one level), but each count stays two
  to three times the homogeneous one. A subdomain that straddles a soft layer has more nearly free
  motions than the three rigid ones: the stiff part can slide on the soft part. Those motions are the
  slow modes left over.
- **GenEO closes the gap.** `jno.space.geneo()` (Spillane et al., *Numer. Math.* 126, 2014) finds those
  motions per subdomain. It solves a local eigenproblem between the subdomain's own operator, assembled
  over its cells with free edges where it is cut, and its block of the global operator, and keeps the
  eigenvectors below `tau = 0.2`. It needs the Schwarz spec it is combined with, so it gets the same
  subdomains:

  ```python
  G = jno.precond.coarse(jno.space.geneo(), parts=S)
  fem.solve(linear=jno.solve.cg(tol=1e-8), precond=G >> S >> G)
  ```

  It takes 26, 38 and 40 iterations at contrast $10^4$, against 23, 39 and 36 for the homogeneous block.
  The contrast no longer costs anything. In a homogeneous subdomain it keeps little more than the three
  rigid modes; in one that straddles a soft layer it keeps the sliding motions too.
- **Every solve is checked.** Each one is compared with the sparse-direct `lu` answer. The counts vary
  by a few iterations from run to run on a GPU, because the order of floating-point sums changes.

## Result

![Top: the block's mesh coloured by the 64 METIS subdomains, with black lines at the layer interfaces; the subdomains cut across every interface. Bottom: CG iterations against 4, 16 and 64 subdomains for one-level Schwarz (dashed), Schwarz with the rigid-body coarse space (solid) and with GenEO (dotted, squares), for the homogeneous block (blue) and contrast 1e4 (red).](/jNO/assets/coarse_spaces_layered_elasticity.png)

METIS cuts the block into compact subdomains with no regard for the layers (top), so most subdomains hold
both materials. Without a coarse space the iterations grow steadily with the number of subdomains
(dashed). The rigid-body modes flatten that growth for both materials (solid), but the high-contrast
curve sits well above the homogeneous one. With GenEO (dotted, squares) the two materials lie on one curve.

See [Coarse spaces and deflation](../../solvers.md#coarse-spaces-and-deflation-jnoprecondcoarse) for
every form a space can take and every way to combine `coarse` with another preconditioner.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/coarse_spaces_layered_elasticity.py:code"
```
