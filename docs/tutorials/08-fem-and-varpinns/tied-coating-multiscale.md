# Thermal Barrier Coating — two meshes, one solve

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/tied_coating_multiscale.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

A thin ceramic coating ($L = 0.05$, $k = 1$) sits on a thick metal substrate ($L = 1.0$, $k = 20$).
A layer **20× thinner**, of a material **20× less conductive**, carries the *same* temperature drop
as the entire substrate beneath it — because conduction in series adds thermal resistances $L/k$,
and here both contribute $0.05$.

With a steady flux $q$ injected at the top and $T = 0$ held at the base, the exact 1-D solution is
piecewise linear with a kink at the interface:

$$T_\text{interface} = \frac{q L_\text{sub}}{k_\text{sub}} = 0.050,
\qquad
T_\text{top} = T_\text{interface} + \frac{q L_\text{film}}{k_\text{film}} = 0.100 .$$

So the coating — 5 % of the thickness — is responsible for **half** the temperature rise. That is
what makes it a barrier, and it is also what makes it hard to mesh: all the action is inside a sliver.

## Why two meshes

Resolving the gradient through the coating needs several elements across $0.05$, so $h \approx
0.0125$. The substrate is happy at $h \approx 0.12$, ten times coarser. One conforming mesh cannot
honour both at the shared interface — it has to pick a single size there and compromise.

!!! measured "What one mesh actually gives you, on this geometry"
    | | coating $h$ (asked 0.0125) | substrate $h$ (asked 0.12) |
    |---|---|---|
    | `conforming=True` | 0.0227 | 0.0995 |
    | `conforming=False` | **0.0119** | **0.1084** |

    Fragmenting the two bodies into one mesh under-resolves the layer you care about *and*
    over-refines the bulk you do not.

Meshing each body independently and gluing them with a tie `u(A) - u(B)` gets both right. The two
interface surfaces then carry different node layouts, so `jno.fem` couples them with a **mortar**
(integrated) constraint rather than node-to-node matching — see
[Tying two boundaries](../../fem/boundary-conditions.md).

## The term list

Two bodies, each meshed at its own size; `conforming=False` skips the fragment, so the shared surface
exists twice — once per body — and each is meshed independently.

```python
d = jno.Shape.regions(
    substrate=jno.Shape.rect(0.0, 0.0, 1.0, L_SUB, size=H_SUB),
    coating=jno.Shape.rect(0.0, L_SUB, 1.0, L_SUB + L_FILM, size=H_FILM),
    conforming=False,
).domain()

fem = jno.fem([
    K_SUB  * (Ts.x * ps.x + Ts.y * ps.y),    # steady conduction in the metal substrate
    K_FILM * (Tc.x * pc.x + Tc.y * pc.y),    # ... and in the ceramic coating
    -Q * phi.bind(x=xt, y=yt),               # flux in at the top (a natural BC)
    T(*a) - T(*b),                           # glue the two bodies
    T(xb, yb) - 0.0,                         # T = 0 at the base
])
```

One conduction term per material region — each integrates over that region's cells only. The tie is
one more entry in the same list.

!!! danger "In `u(A) - u(B)`, the first argument must be the finer side"
    `A` is the **secondary**: its interface DOFs are eliminated in favour of an interpolation from
    the main side. Put the coarse side first and the fine mesh's resolution at the interface is
    thrown away. Measured here:

    | secondary | interface nodes | $T_\text{interface}$ | error |
    |---|---|---|---|
    | `coating` (finer) | 81 | **0.05000** | exact |
    | `substrate` (coarser) | 10 | 0.05531 | 10.62 % |

    Nothing raises — both orders assemble and solve.

## Result

![Temperature against height. The FEM nodes from both meshes lie on the exact piecewise-linear solution, which kinks at the interface; the shaded coating band spans only 5% of the height but half the temperature range.](/jNO/assets/tied_coating_multiscale.png)

```text
Coating (tied, two mesh resolutions): dofs=620
  T at interface  FEM=0.05000  exact=0.05000
  T at top        FEM=0.10000  exact=0.10000
  the 0.05-thick coating carries 50% of the temperature rise
```

Both probes hit the series-resistance solution to five decimals, across a tied interface whose two
sides have different node counts. The script asserts both to within 2 %.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/tied_coating_multiscale.py:code"
```
