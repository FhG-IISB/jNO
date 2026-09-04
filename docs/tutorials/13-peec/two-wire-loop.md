# Two-Wire Loop — resistance and inductance with nothing meshed

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/13_peec/01_two_wire_loop.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

The first thing to know about `jno.peec` is what it does **not** do: it never meshes the air. A
partial-element method discretises only the metal, and the field between conductors is carried by
the partial inductances coupling one filament to another. The input is a conductor and a port; the
output is a circuit.

The conductor here is a hairpin — out along one wire, across the far end, back along the other — and
the port is the pair of open ends.

## The whole problem statement

```python
loop = jno.Shape.line(route, r=RAD, size=2e-3).attach(sigma=CU).name("loop")
sol  = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=0.0).solve()
```

Geometry carries the material, the constraint list says only what is impressed on the terminals, and
`solve()` takes solver choices only. Nothing about the physics is a solve argument.

## Which inductance a formula reports

Two closed forms say what the answer must be, and they behave very differently.

**Resistance** is exact and unforgiving. At DC the current fills the section, so $R = \rho\,\ell/A$
over the whole routed length, the far-end link included. jNO reproduces it to twelve digits — there
is no modelling freedom in it, which makes it the right thing to check first.

**Inductance** needs care, and the care is the lesson. The textbook two-wire line carries

$$\frac{L_\text{ext}}{\ell} = \frac{\mu_0}{\pi}\,\operatorname{acosh}\!\frac{D}{2a}$$

but that is the **external** inductance — the flux in the air between the wires. `sol.L` is the
total magnetic energy, so it also holds the flux *inside* the copper, which for a round wire
carrying a uniform DC current adds $\mu_0/8\pi$ per unit length per wire:

| compared against | result |
|---|---|
| external term alone | **+8.7 %** — looks like an error |
| external + internal | **−2.0 %** — the finite length |

The 8.7 % is not an error bar. It is a **different quantity**, and knowing which one a formula
reports is most of the work in validating an inductance. The remaining 2 % is the closed form having
no ends while a 100 mm hairpin has two.

## Full script

```python
--8<-- "tutorial_examples/13_peec/01_two_wire_loop.py:code"
```
