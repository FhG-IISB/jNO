# Cored Transformer — an open secondary and a lossy core

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/13_peec/03_cored_transformer.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

Everything in the first two tutorials was conductors in air. A region that declares `mu_r` instead
of — or as well as — `sigma` is a **magnetic material**, and it is solved for: the core becomes a
second family of elements on the shared cell grid, carrying magnetisation currents that couple back
into the circuit.

Nothing is switched on. **What a region carries decides what it is:**

```python
core = jno.Shape.box(...).attach(mu_r=2000 - 200j).name("core")   # complex ⇒ lossy
turn = jno.Shape.box(...).attach(sigma=CU).name("pri")            # metal
```

## An open terminal has a voltage

The secondary carries no current by construction, so `current` and the port impedance say nothing
about it — the induced voltage lives in the nodal potentials, and `sol.voltage` is how they are read:

```python
jno.peec([
    v(*at("P0")) - v(*at("P1")) - 1.0,   # drive the primary with one volt
    i(*at("S0")) - 0.0,                  # the secondary is OPEN — no current
    v(*at("S1")) - 0.0,                  # a reference to measure the induced volts against
], freq=1e5).solve().voltage("S0", "S1")
```

With one turn on each limb, the ideal-transformer law appears on its own — and gets closer as the
permeability rises, because a better core leaks less flux:

| $\mu_r$ | $L_\text{pri}$ | $V_\text{sec}/V_\text{pri}$ |
|---:|---:|---:|
| 200 | 207 nH | 0.8115 |
| 2 000 | 1 803 nH | 0.9754 |
| 20 000 | 17 760 nH | **0.9953** |

Nothing in the input imposes a turns ratio. It is the geometry.

## A complex permeability is a lossy core

The imaginary part of $\chi = \mu_r - 1$ is the lossy component of the magnetisation, exactly as a
complex permittivity carries dielectric loss. It comes back from `dissipation()` under `.mu_r` —
named for the property that caused it — while the ohmic loss stays under `.sigma`.

The check is **power balance**, which needs no reference value and no volume. Driven by one volt a
passive network takes in $\mathrm{Re}(1/Z)$, and whatever `joule` does not account for is the core:

| $\mu_r$ | in | copper | core |
|---|---:|---:|---:|
| $2000$ | 7.698e-4 W | 7.698e-4 W | **0** |
| $2000-200j$ | 8.655e-2 W | 7.623e-4 W | **8.579e-2 W** |

A real permeability is lossless *exactly*, not approximately. A lossy one dominates the copper by
two orders of magnitude here.

!!! note "`dissipation()` is a density, on purpose"
    It returns W/m³ per unit of the discretisation's own summed element volume, because a heat
    source is a density — it is shaped for `d.by_region`, not for totalling. `joule` is the total
    that pairs with it on the copper side; the core channel has no such total today, which is why
    the balance above is written with `joule` and `Re(1/Z)`.

A core is **linear** (no saturation), is refused **at DC** — the magnetisation reaches the circuit
only through $j\omega K'$, which vanishes there — and cannot be welded to a `Shape.line`.

## Full script

```python
--8<-- "tutorial_examples/13_peec/03_cored_transformer.py:code"
```
