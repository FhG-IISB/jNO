# Callouts — the docs style

Seven callout types, each with one job. The point is that a reader can tell from the **colour alone**
whether a block is background they can skip, advice they should take, or a trap that will cost them an
afternoon — without reading it first.

Every type works **open** (`!!!`) or **collapsed** (`???`). Collapsed is the default choice for
anything longer than a short paragraph: write as much as you want, and the page stays scannable.

## The seven types

| Type | Use it for | Open or collapsed |
|---|---|---|
| `abstract` | a one-paragraph summary opening a long page | open |
| `note` | neutral context, an adjacent fact | either |
| `tip` | actionable advice — "do it this way" | open |
| `measured` | a benchmarked number | either |
| `fun-fact` | a tangent worth knowing, safe to skip | collapsed |
| `warning` | a footgun — it works, but it will bite | open |
| `danger` | silently wrong, destructive, or a hard trap | **always open** |

!!! abstract "In short"
    Opens a long page with the one thing to remember. If a reader reads only this block, they should
    still leave with the correct headline.

!!! note "Neutral context"
    An adjacent fact that helps but is not advice. The catch-all — which means if you find yourself
    reaching for `note` more than the others combined, one of the specific types is probably right.

!!! tip "Actionable advice"
    A recommendation the reader can act on: prefer `h`-refinement over `order ≥ 2` near a curved
    boundary.

!!! measured "A benchmarked number"
    Reserved for numbers that were actually measured, with the conditions attached. The persistent
    XLA cache takes a 3-D Poisson build at 27,833 nodes from **4.75 s to 2.22 s**; the repeat build
    from 2.48 s to 1.51 s.

??? fun-fact "A tangent worth knowing"
    Collapsed by default, because it is optional by definition. The `±1` diffraction pair is
    degenerate at normal incidence, so the modal eigensolve splits it at its own floor — around
    2e-8, and more Fourier orders do not lower it.

!!! warning "A footgun"
    It works, but it will bite you. A parameter on a transient mass term is assembled once and would
    be silently frozen — so it fails loud instead.

!!! danger "Silently wrong"
    Never collapse this one. The `2πr` measure is exact for scalars and **wrong for vectors**, and
    the assembler cannot tell the difference from a legitimate radial coefficient, so nothing raises.

## Collapsing

Any type collapses by swapping `!!!` for `???`. Add `+` (`???+`) to render it open but foldable.

```markdown
??? measured "cuDSS vs SuperLU on the Newton re-factor"
    Everything here is hidden until the reader asks for it.
```

Use it for derivations, benchmark tables, rationale, and anything that answers *why* rather than
*what*. A page should read completely with every collapsed block shut.

!!! danger "One thing never collapses"
    A `danger` block. If a reader can lose a result by not knowing something, it does not go behind a
    click.

## Rules of thumb

- **Two open callouts in a row is a smell.** Merge them, or collapse one.
- **A callout is not a paragraph with a border.** If it reads as continuous prose with what precedes
  it, it is prose — take the box off.
- **Every `measured` block names its conditions.** A number without a mesh size, a device, or a
  problem is not a measurement.
- **Prefer a table inside the callout** to a list of numbers in a sentence.
