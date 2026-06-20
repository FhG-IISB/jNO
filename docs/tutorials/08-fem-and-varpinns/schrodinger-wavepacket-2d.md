# 2D Schrödinger: a wave packet scattering off a barrier (complex, unitary)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/schrodinger_wavepacket_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

The time-dependent Schrödinger equation in 2-D — a genuinely **complex** wavefunction evolving in
time:

$$i\,\frac{\partial \psi}{\partial t} = \hat H\,\psi,\qquad \hat H = -\tfrac12\,\Delta + V(\mathbf x).$$

A Gaussian wave packet (its momentum carried by the phase $e^{i\mathbf k\cdot\mathbf x}$) is launched
at a tall, thin potential barrier; part **tunnels through** and part **reflects**, with interference
fringes where the incoming and reflected waves overlap.

## Real `M`, `H` from jno.fem — the `i` is in the time stepping

The spatial operators are real and symmetric: the mass $M$ and the Hamiltonian
$H = \tfrac12\,\text{(stiffness)} + V\,\text{(mass)}$. jno.fem assembles both; the imaginary unit
enters only through time integration.

```python
V = V0 * jno.np.exp(-((xi - xbar) ** 2) / (2 * 0.04**2))          # a tall, thin barrier
block = jno.fem([ui.t * vi + 0.5 * (ui.x*vi.x + ui.y*vi.y) + V*(u*vi), u(ci[0], ci[1]) - 0.0]).operator
M, H = dense(block.M), dense(block.A)                              # real mass + Hamiltonian
```

## Bring your own *unitary* integrator (Crank–Nicolson)

Schrödinger evolution is **unitary** — $\int|\psi|^2$ is conserved exactly. The default backward-Euler
is strongly *dissipative* for it (the packet would fade away), so we bring our own **Crank–Nicolson**
stepper, the Cayley transform of $H$, which conserves the norm to machine precision:

$$\Big(M + \tfrac{i\,\Delta t}{2}H\Big)\psi_{n+1} = \Big(M - \tfrac{i\,\Delta t}{2}H\Big)\psi_n.$$

```python
P = jnp.linalg.solve(M + 0.5j*dt*H, M - 0.5j*dt*H)   # CN propagator, factored once
psi = psi0                                            # complex Gaussian packet, exp(i k x)
for _ in range(nsteps):
    psi = P @ psi                                     # one unitary step (a matvec)
```

## The result

![Animation of |psi|^2: a Gaussian wave packet moves right, strikes the dashed barrier, and splits —
a reflected part forms vertical interference fringes on the left while a transmitted part continues
to the right.](/jNO/assets/schrodinger_wavepacket_2d.gif)

The packet hits the barrier (dashed line) and splits: ~⅔ reflects (the vertical fringes are the
incoming and reflected waves interfering) and ~⅓ tunnels through — even though the barrier sits
*above* the packet's mean energy.

## What to notice

- **Complex is native:** the packet's momentum is the phase $e^{i\mathbf k\cdot\mathbf x}$; `jno.np`
  carries the complex arithmetic, and `psi` is `complex128` throughout.
- **The right integrator matters** — exactly the lesson from the
  [diffrax heat spreader](transient-diffrax-heat-spreader.md): the default backward-Euler is
  dissipative, so for a *unitary* problem we supply Crank–Nicolson and the norm is conserved to
  rel-error $<10^{-3}$.
- **Verified by physics, not an analytic solution:** the norm is conserved, the packet genuinely
  splits (both reflected and transmitted fractions $>0.1$), and $\psi$ is genuinely complex.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/schrodinger_wavepacket_2d.py"
```
