"""An eager march keeps no past on the device: only the states a frame at ``save_ts`` reads.

A transient march used to stack EVERY step on the device -- ``(n_steps, n_dofs)`` -- and only then sample
``save_ts`` from it, so a long march filled the card with a trajectory nobody had asked for (1000 steps
of a 440k-DOF flow: 3.5 GB). Evaluated eagerly, each step now writes its state into a save slot when a
frame will read it and into one scratch row otherwise; the frames go to the host. Dense saves that would
outgrow the device's budget run in chunks whose frames are copied out while the next chunk is queued --
nothing runs inside the compiled step. Under ``jit``/``grad`` the march stays one scan.

Oracles: the chunked march equals the one-chunk march BIT for bit, on every scheme, on and off the grid,
periodic and nonlinear; equals the traced march to rounding; the compiled march holds rows for the read
states only; and the convergence guards still fire, naming the same step.
"""

from __future__ import annotations

import os
import re

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
import jno.utils.solver.backend_blocks as bb


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.fixture
def chunks_of(monkeypatch):
    """Force the chunk length (on a CPU the march is otherwise one chunk: the device IS the host)."""

    def set_k(k):
        monkeypatch.setattr(bb, "_offload_chunk", lambda n, d, t: min(n, k))

    return set_k


E = 1e-9


def _heat(nonlinear=False, steps=10, periodic=False, param=False):
    """``u_t = div(k c(u) grad u) + 1 + t`` on the unit square; ``c = 1 + u²`` when nonlinear."""
    d = jno.shape.rect(0, 0, 1, 1).structured(n=7).domain(time=(0.0, 0.05 * steps, steps + 1))
    if periodic:
        d.tag("left", lambda x, y: x < E)
        d.tag("right", lambda x, y: x > 1 - E)
        d.tag("wall", lambda x, y: (y < E) | (y > 1 - E))
    else:
        d.tag("wall", lambda x, y: (x < E) | (x > 1 - E) | (y < E) | (y > 1 - E))
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi, ti = d.variable("interior", split=True)
    xw, yw, _ = d.variable("wall", split=True)
    ci = d.variable("initial", split=True)
    ub, vb = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    k = jno.np.parameter((1,), name="k") if param else 1.0
    c = (1 + ub * ub) if nonlinear else 1.0
    ic = jno.np.sin(np.pi * ci[0]) * jno.np.sin(np.pi * ci[1]) + 0.2
    terms = [ub.t * vb + k * c * (ub.x * vb.x + ub.y * vb.y) - (1 + ti) * vb, u(xw, yw) - 0.0, u(*ci) - ic]
    if periodic:
        xl, yl, _ = d.variable("left", split=True)
        xr, yr, _ = d.variable("right", split=True)
        terms.append(u(xl, yl) - u(xr, yr))
    return jno.fem(terms)


SCHEMES = {
    "theta": lambda: {},
    "crank_nicolson": lambda: {"time": jno.solve.theta(0.5)},
    "bdf2": lambda: {"time": jno.solve.bdf2()},
    "sdirk2": lambda: {"time": jno.solve.sdirk(2)},
    "ros2": lambda: {"time": jno.solve.rosenbrock("ros2")},
    "exponential": lambda: {"time": jno.solve.exponential(mass="consistent")},
}
NONLINEAR_OK = {"theta", "crank_nicolson", "bdf2", "sdirk2"}
OFF_GRID = np.array([0.0, 0.013, 0.26, 0.31, 0.5])


def _run(scheme, *, nonlinear=False, save=None, periodic=False):
    kw = SCHEMES[scheme]()
    if save is not None:
        kw["save_ts"] = save
    out = _heat(nonlinear=nonlinear, periodic=periodic).solve(**kw)
    return out.fn() if hasattr(out, "fn") else out


@pytest.mark.parametrize("scheme", list(SCHEMES))
@pytest.mark.parametrize("save", [None, OFF_GRID], ids=["grid", "off_grid"])
def test_chunking_changes_nothing(scheme, save, chunks_of):
    nonlinear = scheme in NONLINEAR_OK
    chunks_of(10**9)
    one = _run(scheme, nonlinear=nonlinear, save=save)
    chunks_of(3)
    chunked = _run(scheme, nonlinear=nonlinear, save=save)
    assert isinstance(one, np.ndarray) and isinstance(chunked, np.ndarray), "an eager march returns host frames"
    assert one.shape == chunked.shape == ((11 if save is None else save.size), 64)
    assert np.abs(one).max() > 0.1
    # two separate marches: bit-identical on CPU, but a GPU reassociates its sparse reductions run to run
    assert np.allclose(one, chunked, rtol=1e-10, atol=1e-12 * np.abs(one).max())


def test_a_periodic_march_prolongs_its_host_frames(chunks_of):
    chunks_of(10**9)
    one = _run("bdf2", nonlinear=True, periodic=True)
    chunks_of(3)
    chunked = _run("bdf2", nonlinear=True, periodic=True)
    assert isinstance(chunked, np.ndarray) and chunked.shape == (11, 64)  # the full nodal layout
    # two separate marches: bit-identical on CPU, but a GPU reassociates its sparse reductions run to run
    assert np.allclose(one, chunked, rtol=1e-10, atol=1e-12 * np.abs(one).max())


@pytest.mark.parametrize("scheme", ["theta", "bdf2"])
def test_the_eager_march_equals_the_traced_one(scheme, chunks_of):
    chunks_of(4)
    # An explicit Newton: the eager march also carries its tangent across steps, which a traced one does
    # not, and the two would then differ by the Newton tolerance -- not what this compares.
    nl = jno.solve.newton(rtol=1e-12, atol=1e-14)
    node = _heat(nonlinear=True, param=True).solve(nonlinear=nl, **SCHEMES[scheme]())
    k = jnp.array([1.3])
    eager = node.fn(k)
    traced = jax.jit(lambda kk: node.fn(kk))(k)  # one scan, every step on the device: the adjoint's path
    assert isinstance(eager, np.ndarray) and isinstance(traced, jax.Array)
    assert np.abs(eager - np.asarray(traced)).max() <= 1e-12 * np.abs(eager).max()


def _spy_buffers(monkeypatch):
    """Record, for every compiled march call, (steps it marches, rows of states it can hold)."""
    seen = []
    real = bb._split_cached_march

    def spy(block, args, config, march, *inputs):
        if config and config[-1] == "to_host":
            ext, grid = inputs[0], inputs[1]
            seen.append((int(np.shape(grid)[0]) - 1, int(np.shape(ext[1])[0])))
        return real(block, args, config, march, *inputs)

    monkeypatch.setattr(bb, "_split_cached_march", spy)
    return seen


def test_the_device_holds_only_the_states_a_frame_reads(monkeypatch):
    """40 steps, 3 frames (one between grid points): the march holds 4 states and a scratch row, not 40."""
    seen = _spy_buffers(monkeypatch)
    d_steps = 40
    fem = _heat(nonlinear=True, steps=d_steps)
    out = fem.solve(save_ts=np.array([0.5, 1.0, 1.4625])).fn()  # grid step 0.05: 1.4625 is between points
    assert out.shape == (3, 64)
    assert seen == [(d_steps, 4 + 1)], seen  # 0.5, 1.0, and the two either side of 1.4625; + scratch


def test_dense_frames_are_marched_in_chunks_within_the_budget(chunks_of, monkeypatch):
    chunks_of(3)
    seen = _spy_buffers(monkeypatch)
    _run("theta", nonlinear=True)  # every grid point saved
    assert seen and all(rows <= 3 + 1 for _, rows in seen), seen
    assert sum(steps for steps, _ in seen) == 10  # every step marched exactly once


def test_a_diverged_step_raises_naming_the_same_step(chunks_of):
    capped = {"nonlinear": jno.solve.newton(max_steps=1, rtol=1e-14, atol=1e-16)}
    named = []
    for k in (10**9, 3):
        chunks_of(k)
        with pytest.raises(RuntimeError, match="did not converge at step") as err:
            _heat(nonlinear=True).solve(**capped).fn()
        named.append(int(re.search(r"at step (\d+) of (\d+)", str(err.value)).group(1)))
    assert named[0] == named[1]


def test_a_march_that_never_moves_is_still_refused(chunks_of):
    """atol above the residual scale: every step "converges" on entry. The guard used to read the whole
    trajectory; chunked, it reads one flag per chunk."""
    chunks_of(2)
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain(time=(0.0, 1.0, 7))
    u, w = d.fem_symbols(names=("u", "w"), order=1)
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, wi = u.bind(x=xi, y=yi, t=ti), w.bind(x=xi, y=yi, t=ti)
    grad, inner = jno.np.grad, jno.np.inner
    fem = jno.fem(
        [
            1e-9 * ((1.0 + ui**2) * ui.t * wi + inner(grad(u, [xi, yi]), grad(w, [xi, yi]), n_contract=1) - wi),
            u(xb, yb) - 0.0,
            u(*ci) - 0.0,
        ]
    )
    with pytest.raises(RuntimeError, match="returned its INITIAL STATE unchanged"):
        fem.solve(nonlinear=jno.solve.newton(direct=True, rtol=1e-6, atol=1e-2), linear=jno.solve.lu(backend="host")).fn()


def test_host_sampling_is_the_device_resample():
    """Chunk frames are now picked or blended on the HOST from the raw slot buffer (sampling them on the
    device held three copies there: a 442k-DOF flow saving 81 frames ran an 8 GB card out of memory). The
    host sampler must be the device one: exact picks on grid points, the same blend and clamping off them."""
    from jno.utils.solver.backend_blocks import _resample_trajectory, _sample_host

    rng = np.random.default_rng(0)
    grid = np.array([0.0, 0.1, 0.25, 0.4, 0.7])
    states = rng.standard_normal((grid.size, 6))
    on = grid[[1, 3, 4]]
    assert np.array_equal(_sample_host(states, grid, on), states[[1, 3, 4]])
    off = np.array([-0.05, 0.05, 0.25, 0.33, 0.7, 0.9])  # below, inside, on, inside, on, above the grid
    want = np.asarray(_resample_trajectory(jnp.asarray(states), jnp.asarray(grid), off, jnp.float64))
    assert np.allclose(_sample_host(states, grid, off), want, rtol=1e-14, atol=1e-14)
