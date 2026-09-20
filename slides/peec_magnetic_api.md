# What the API would look like with magnetic materials

**Status:** a proposal. `.attach(mu_r=...)` and the readouts marked *(new)* do not exist yet; everything
else in these snippets is today's `jno.peec`. Nothing here has been run.

## The design rule

`Shape.attach(**props)` already takes arbitrary named properties, so declaring a magnetic material
needs no new mechanism. **What a region carries decides what it is** — there is no mode flag, no
solver argument, and no second entry point:

| a region attaching | is |
|---|---|
| `sigma=` | a conductor — today's case |
| `mu_r=` | a magnetic material that does not conduct: ferrite, a powder core |
| both | a conducting magnetic material: laminated steel, a lossy core |
| neither | not metal; it is not discretised |

The four constraint forms are **unchanged**. A port is electric whether or not there is a core in the
picture, so nothing a user already knows stops being true.

## 1. A gapped inductor

The smallest realistic magnetic problem: one winding through a gapped ferrite core.

```python
import jax
jax.config.update("jax_enable_x64", True)
import jno

CU, MU_R = 5.8e7, 2400.0                      # copper; N87 ferrite at 100 kHz
GAP = 0.5e-3

core = (jno.Shape.box(0, 0, 0, 0.020, 0.020, 0.010, size=(0.5e-3,) * 3)
        - jno.Shape.box(0.005, 0.005, -1e-3, 0.015, 0.015, 0.011)      # the window
        - jno.Shape.box(0.008, -1e-3, 0.004, 0.012, 0.021, 0.004 + GAP))  # the gap
core = core.attach(mu_r=MU_R).name("core")     # no sigma: a ferrite does not conduct

turn = (jno.Shape.line([(0.010, 0.002, 0.002), (0.010, 0.002, 0.008),
                        (0.010, 0.018, 0.008), (0.010, 0.018, 0.002)], r=0.4e-3, size=1e-3)
        .attach(sigma=CU).name("turn"))

d = (core + turn).domain()
d.tag("A", lambda x, y, z: (y < 0.003) & (z < 0.003))
d.tag("B", lambda x, y, z: (y > 0.017) & (z < 0.003))

i, v = d.peec_symbols()
at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]

sol = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=100e3).build().solve()
print(f"L = {float(sol.L) * 1e9:.1f} nH    R = {float(sol.R) * 1e3:.2f} mOhm")
```

The only line that differs from a coreless model is `.attach(mu_r=MU_R)`. Remove it and the same
script solves the air-cored inductor — which is the comparison a designer wants anyway.

## 2. A transformer — two ports, and the coupling between them

Mutual inductance needs no new constraint vocabulary: an **open** secondary is already sayable as
`i(S1) - 0`, and the induced voltage is read off the terminal.

```python
pri = winding(turns=8, r_in=0.006).attach(sigma=CU).name("pri")
sec = winding(turns=2, r_in=0.009).attach(sigma=CU).name("sec")
core = shell_core().attach(mu_r=MU_R).name("core")

d = (core + pri + sec).domain()
for nm, fn in ports.items():
    d.tag(nm, fn)
i, v = d.peec_symbols()
at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]

emag = jno.peec([
    v(*at("P0")) - v(*at("P1")) - 1.0,        # drive the primary
    i(*at("S0")) - 0.0,                       # secondary open, so it carries no net current
], freq=100e3).build()

sol = emag.solve()
n = sol.voltage("S0") - sol.voltage("S1")     # (new) induced volts across the open secondary
print(f"turns ratio {abs(complex(n)):.3f}, magnetising L = {float(sol.L) * 1e6:.2f} uH")
```

**The one genuinely new readout is `sol.voltage(terminal)`.** The solve already computes nodal
potentials — `solve_network` returns them and the front door discards them — so this is exposing a
value that exists, not computing a new one. Everything else composes from the four forms.

## 3. Coupling to heat — where the case for doing this in jNO actually lives

`sol.dissipation()` already returns `{region: W/m³}` shaped for `d.by_region`, which is a heat source
term. With a core, the core's own loss joins the same dictionary, and the fixed point closes
**inside one differentiable program**:

```python
emag = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=100e3).build()

def step(T):
    """One electro-thermal pass: hot copper conducts worse, and a hot core is lossier."""
    sigma = {"turn": CU / (1 + 3.9e-3 * (T - 293.0))}
    q = emag.solve(sigma=sigma).dissipation()
    heat = d.k * grad(T_, coords) . grad(s, coords) - d.by_region(q, default=0.0) * s
    return jno.fem([heat, T_(d.boundary) - 293.0]).solve()

T = jno.core(step).fixed_point(T0, tol=1e-3)
```

pypeec cannot do this at all — its solve is not differentiable and not composable with a PDE solver.
Ansys can, through a co-simulation, but not inside one gradient.

## 4. What you can then optimise

Because the whole chain is differentiable, the design variable can sit anywhere in it:

```python
mu = jno.np.parameter(2400.0).optimizer(...)        # a material choice
sig = jno.np.parameter(CU * rho).optimizer(...)     # a density field: where copper should be

def objective(p):
    sol = emag.solve(sigma=resolve(p))
    return float(sol.L) + 1e3 * sol.joule           # loop inductance against loss
```

Realistic targets: **bond-wire routing** against loop inductance, **winding geometry** against AC
loss, **die placement** against junction temperature, **gap length** against saturation margin.

**One honest limit:** a *conductivity* may be traced, and a wire's route and gauge may be traced, but
a **lattice shape may not** — so `GAP` above is a rebuild-per-point sweep, not a gradient. Making
solid geometry differentiable is a separate piece of work from magnetic materials.

## Summary of new public surface

| | |
|---|---|
| `.attach(mu_r=...)` | no new mechanism — `attach` is already generic |
| `sol.voltage(terminal)` | **the only new readout**; the value already exists internally |
| `sol.dissipation()` | unchanged signature, core regions simply appear in it |
| constraint forms | **unchanged** — still four, still electric |
| `solve()` arguments | **unchanged** |

That is one new keyword argument and one new readout for a whole class of problem, which is the bar
a new capability should clear.
