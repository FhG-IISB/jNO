# --8<-- [start:code]
"""Full-waveform inversion: recover a wave speed from a trajectory through a 2nd-order ``u_tt`` solve.

The inverse sibling of the vibrating membrane. A wave obeys

    u_tt = c² Δu ,    u = 0 on the boundary,    u(t=0) = sin(πx) sin(πy) ,   u_t(t=0) = 0 ,

and the medium's speed ``c²`` is *unknown*. Given the observed displacement history ``u_obs(t)`` (a
"seismogram"), recover ``c²`` by differentiating the **time integration itself**: for a second-order
weak form ``fem.solve()`` returns the trajectory ``u(save_ts)`` marched with the energy-conserving
trapezoidal (θ=½) rule, and the gradient flows through every step of the augmented ``[u, v=u_t]`` block
back to the parameter. ``crux`` then fits it to the data. This is the mechanism behind full-waveform
inversion and elastography — the same call recovers a density on the ``u_tt`` term or a shear modulus
in a vector (elastodynamic) form.

A wrong speed makes the wave oscillate at the wrong frequency, so the misfit is sharply informative;
the recovered trajectory lands back on top of the data. (Second-order soft modes need float64 — see
the ringing-cantilever tutorial — so we opt into x64 up front.)
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402
from jno.utils.solver.backend_blocks import _block_time_grid, _default_transient_integrate  # noqa: E402

c2_true, c2_guess = 2.0, 1.0  # the unknown wave speed² to recover, and a deliberately wrong start

# --- forward wave u_tt = c² Δu on a clamped square, plucked into its fundamental mode ---
d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.14, time=(0.0, 1.5, 60))
u, phi = d.fem_symbols()
xi, yi, ti = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
xi0, yi0, ti0 = d.variable("initial", split=True)
ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
ui0 = u.bind(x=xi0, y=yi0, t=ti0)
u0 = jno.np.sin(np.pi * xi0) * jno.np.sin(np.pi * yi0)
c2 = jno.np.parameter((1,), name="c2")  # the unknown coefficient
fem = jno.fem([ui.tt * vi + c2 * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(xi0, yi0) - u0, ui0.t - 0.0])

# --- the "observed" seismogram: the forward wave at the true speed (the data) ---
blk = fem.operator
ts = np.asarray(_block_time_grid(blk))
u_obs = np.asarray(_default_transient_integrate(blk, {"c2": c2_true}, ts))

# --- recover c² from the data through the differentiable transient solve ---
c2.dtype(jnp.float64)
c2.initialize(jax.nn.initializers.constant(c2_guess))  # start at the wrong speed
c2.optimizer(optax.adam(5e-2))
crux = jno.core([(fem.solve() - u_obs).mse], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
crux.solve(220)
rec = float(np.asarray(crux.eval([c2])).reshape(-1)[0])

print(f"\nFull-waveform inversion (u_tt = c² Δu):  recovered c² = {rec:.4f}  (truth {c2_true})")
print(f"  started at c² = {c2_guess}   ->   rel-err = {abs(rec - c2_true) / c2_true:.2%}")
assert abs(rec - c2_true) / c2_true < 0.02, f"wave speed not recovered: c²={rec:.4f} (truth {c2_true})"
# --8<-- [end:code]

# --- figure: the receiver seismogram — data vs the wrong start vs the recovered fit ---
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

n = fem.offsets[1]
pts = np.asarray(fem.points)
ci = int(np.argmin(np.sum((pts - 0.5) ** 2, axis=1)))  # a receiver near the (0.5, 0.5) antinode
u_start = np.asarray(_default_transient_integrate(blk, {"c2": c2_guess}, ts))
u_rec = np.asarray(_default_transient_integrate(blk, {"c2": rec}, ts))

plt.rcParams.update({"savefig.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 10})
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.plot(ts, u_obs[:, ci], lw=2.6, color="0.75", label=f"observed (c²={c2_true})")
ax.plot(ts, u_start[:, ci], ":", lw=1.4, color="C3", label=f"start (c²={c2_guess}, wrong speed)")
ax.plot(ts, u_rec[:, ci], "--", lw=1.4, color="C0", label=f"recovered (c²={rec:.3f})")
ax.set_xlabel("time")
ax.set_ylabel("receiver displacement")
ax.set_title("Full-waveform inversion: fitting the wave speed to the seismogram")
ax.legend(loc="upper right", frameon=False, fontsize=8)
ax.margins(x=0)
assets = Path(__file__).resolve().parents[2] / "assets"
assets.mkdir(exist_ok=True)
fig.savefig(assets / "inverse_wave_speed_2d.png")
print(f"  saved figure -> {assets / 'inverse_wave_speed_2d.png'}")
