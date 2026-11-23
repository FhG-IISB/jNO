# Hot gas in a thick-walled vessel (three interface conditions)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/15_coupling/hot_gas_vessel.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

A hot gas at rest in a long cylinder (`jno.fvm`: the Euler equations with heat conduction) cools into the thick
wall around it (`jno.fem`: transient conduction and plane-strain thermo-elasticity). The two meet on the bore
through the three kinds of interface condition, each written as for one problem: the temperatures agree, the heat
that leaves the gas enters the wall, and the gas pressure loads the wall.

## Result

Over ten coupling windows the gas gives 0.094 of heat to the wall; its pressure falls from 3.00 to 2.81, the wall
warms to 1.18 at the bore (nondimensional) and the bore moves out by $2.7\times10^{-2}$. The heat the wall takes in
is the heat the gas gave out to 0.3 %. The temperature and flux conditions are a Dirichlet-Neumann pair, iterated
seven to eleven times per window; the pressure is a load. It runs in under two minutes on a CPU.

## The code

```python
--8<-- "tutorial_examples/15_coupling/hot_gas_vessel.py:code"
```

## What to notice

- **A gas that takes a wall temperature.** An inviscid gas has no heat conduction, so a temperature on its wall
  would over-determine it. With conduction in its energy equation it is a diffusion problem for the temperature,
  and the temperature can be prescribed. `jno.fvm` takes one condition per unknown and face, and a condition must
  name one unknown: the temperature is therefore a fifth unknown `T` with the algebraic row `T - p/rho`, beside
  the vector `q` that takes the slip condition.
- **The conduction in divergence form.** In a vector equation the conduction goes into the energy row as the
  divergence of a stacked flux, `stack([0, 0, 0, -k T.x]).d(x, scheme="finite_difference") + ...`. It is then a
  face flux and the gas conserves its energy exactly.
- **Who takes what.** The coupling probes both sides: the gas, the softer one, takes the temperature, and the wall
  takes the heat flux and the pressure (a load, since it differentiates only the wall).
- **Steps of their own.** The wall takes one backward Euler step per window; the gas takes twenty explicit steps
  (`jno.solve.imex`: the fluxes explicit with its CFL substeps, the algebraic temperature solved in every cell).
  The heat flux the wall receives is the gas's mean over the window, so the heat the wall takes in is the heat the
  gas gave out. The gas can also step implicitly (`time={gas: None, ...}`, backward Euler): one step per window,
  about five times its acoustic step limit, gives the same pressure, wall temperature and energy balance.

See [Coupling problems on their own meshes](../../coupling.md).
