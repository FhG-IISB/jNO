"""A nonlinear solve across several devices: the cells split, the answer does not move.

A Jacobian-free Newton has no assembled operator to partition, so its ELEMENT axis is split instead: each
device evaluates the residual's element kernel on its share of the cells into its own partial residual,
and one all-reduce combines them. ``J.v`` is the linearisation of that map and splits the same way, while
the Krylov vectors stay replicated -- so the solver is untouched. A nonlinear MARCH does the same inside
its scan, one split Newton solve per step. Checked here on simulated CPU devices
(in a subprocess: ``XLA_FLAGS`` must precede JAX's start) against a one-device solve, and on the compiled
``J.v`` itself -- answers alone cannot show a split that silently gathers everything back onto every
device. No speed is asserted.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

_INNER = textwrap.dedent(
    """
    import json
    import jax, numpy as np
    jax.config.update("jax_enable_x64", True)
    import jno
    from jno.utils.solver.sharding import element_devices

    inner, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
    dot = lambda a, b: inner(a, b, n_contract=1)
    ddot = lambda a, b: inner(a, b, n_contract=2)

    def reaction(k=None):
        # -lap u + k u^3 = 10, u = 0 on the boundary; 946 cells, which 4 and 8 devices do NOT divide
        d = jno.shape.rect(0, 0, 1, 1, size=0.05).domain()
        u, v = d.fem_symbols()
        c = d.variable("interior", split=True); cb = d.variable("boundary", split=True)
        ui, vi = u.bind(x=c[0], y=c[1]), v.bind(x=c[0], y=c[1])
        kk = 1.0 if k is None else jno.np.parameter((1,), name="k")
        return jno.fem([ui.x * vi.x + ui.y * vi.y + kk * ui**3 * vi - 10.0 * vi, u(cb[0], cb[1]) - 0.0])

    def cavity():
        # steady Navier-Stokes, Taylor-Hood, a regularised lid: two fields, mixed order, a saddle point
        d = jno.shape.rect(0, 0, 1, 1, size=0.12).domain()
        u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
        p, q = d.fem_symbols(names=("p", "q"), order=1)
        x, y, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        U, V, P, Q = (f.bind(x=x, y=y) for f in (u, v, p, q))
        gu, gv = grad(u, [x, y]), grad(v, [x, y])
        return jno.fem([
            dot(dot(gu, U), V) + 0.05 * ddot(gu, gv) - P * trace(gv), Q * trace(gu),
            u(xb, yb)[0] - 16.0 * yb * xb**2 * (1.0 - xb) ** 2, u(xb, yb)[1] - 0.0, p.pin(),
        ])

    def placed(a):
        s = getattr(a, "sharding", None)
        return len(s.device_set) if s is not None else 0

    out = {}
    fem = reaction()
    a, b = fem.solve(), fem.solve(shard=False)
    out["reaction"] = np.asarray(a).tolist()
    out["reaction_devices"], out["opt_out_devices"] = placed(a), placed(b)
    out["opt_out_diff"] = float(np.abs(np.asarray(a) - np.asarray(b)).max())

    # the compiled J.v under the split: its collectives and its per-device scratch
    R = fem._op.residual
    w = jax.numpy.asarray(a) * 0.5
    def compiled(devs):
        with element_devices(devs):
            f = jax.jit(lambda y, z: jax.jvp(lambda s: R(s, {}), (y,), (z,))[1])
            return f.lower(w, w).compile()
    one, many = compiled([]), compiled(jax.devices())
    txt = many.as_text()
    out["all_reduce"], out["all_gather"] = txt.count("all-reduce("), txt.count("all-gather(")
    out["permute"] = txt.count("collective-permute(")
    out["temp_ratio"] = many.memory_analysis().temp_size_in_bytes / one.memory_analysis().temp_size_in_bytes

    # a runtime parameter: the eager solve is jitted and CACHED -- the split must not leak across calls
    fk = reaction(k=True)
    ks = fk.solve(k=1.0)
    kf = fk.solve(k=1.0, shard=False)
    out["param_diff"] = float(np.abs(np.asarray(ks) - np.asarray(a)).max())
    out["param_devices"], out["param_opt_out_devices"] = placed(ks), placed(kf)
    # reverse mode through the split: custom_root's adjoint transposes the split J.v
    loss = lambda kk: jax.numpy.sum(fk.solve(k=kk) ** 2)
    out["grad"] = float(jax.grad(loss)(jax.numpy.asarray([1.3]))[0])

    fc = cavity()
    out["cavity"] = np.asarray(fc.solve()).tolist()
    out["direct"] = np.asarray(fc.solve(nonlinear=jno.solve.newton(direct=True))).tolist()

    # a NONLINEAR march: a Newton solve per step, each residual split inside the scan
    import jno.utils.solver.fem_native as FN
    calls, real = [], FN._sharded_element_add
    FN._sharded_element_add = lambda *a, **k: calls.append(1) or real(*a, **k)
    d = jno.shape.rect(0, 0, 1, 1, size=0.06).domain(time=(0.0, 0.05, 11))
    u, v = d.fem_symbols()
    c = d.variable("interior", split=True); cb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=c[0], y=c[1], t=c[2]), v.bind(x=c[0], y=c[1], t=c[2])
    fm = jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y + 5.0 * ui**3 * vi - 2.0 * vi, u(cb[0], cb[1]) - 0.0,
                  u(ci[0], ci[1]) - jno.np.sin(np.pi * ci[0]) * jno.np.sin(np.pi * ci[1])])
    run = lambda **kw: (lambda s: s.fn() if hasattr(s, "fn") else s)(fm.solve(**kw))
    m = run()
    out["march"], out["march_devices"], out["march_split"] = np.asarray(m).tolist(), placed(m), len(calls)
    del calls[:]
    out["march_opt_out_devices"], out["march_opt_out_split"] = placed(run(shard=False)), len(calls)
    print("RESULT " + json.dumps(out))
    """
)


def _run(n_dev):
    env = {**os.environ, "XLA_FLAGS": f"--xla_force_host_platform_device_count={n_dev}", "JAX_PLATFORMS": "cpu"}
    r = subprocess.run([sys.executable, "-c", _INNER], env=env, capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(next(ln for ln in r.stdout.splitlines() if ln.startswith("RESULT "))[len("RESULT ") :])


@pytest.fixture(scope="module")
def one():
    return _run(1)


def test_one_device_takes_the_plain_path(one):
    assert one["reaction_devices"] <= 1 and one["param_devices"] <= 1
    assert one["march_devices"] <= 1 and one["march_split"] == 0


@pytest.mark.parametrize("n_dev", [2, 4])
def test_the_split_solve_matches_one_device(one, n_dev):
    many = _run(n_dev)
    # Only the summation order moves, so the answers agree to the SOLVER's tolerance, not to round-off: a
    # Newton stopped at 1e-8 on a saddle point with a 1e-10 inner solve moved the cavity's pressure by
    # 1.2e-11 on 4 devices. The reaction problem is SPD and well conditioned, and holds 1e-11.
    np.testing.assert_allclose(many["reaction"], one["reaction"], rtol=0, atol=1e-11, err_msg="reaction")
    np.testing.assert_allclose(many["march"], one["march"], rtol=0, atol=1e-11, err_msg="march")
    for key in ("cavity", "direct"):
        np.testing.assert_allclose(many[key], one[key], rtol=0, atol=1e-9, err_msg=key)
    assert max(abs(x) for x in one["cavity"]) > 0.1, "the cavity must actually flow"
    assert many["reaction_devices"] == n_dev, "the solve did not run across the devices"
    # the compiled J.v: ONE all-reduce, nothing gathered or shifted, and each device's scratch shrinks
    assert many["all_reduce"] == 1, many
    assert many["all_gather"] == 0 and many["permute"] == 0, many
    assert many["temp_ratio"] < 1.5 / n_dev, many
    # opting out, and a cached parametric solve on either side of it
    assert many["opt_out_devices"] == 1 and many["opt_out_diff"] < 1e-11
    assert many["param_devices"] == n_dev and many["param_opt_out_devices"] == 1
    assert many["param_diff"] < 1e-11
    assert abs(many["grad"] - one["grad"]) < 1e-10 * abs(one["grad"]), (many["grad"], one["grad"])
    # the march split its residual inside the scan, and opting out did not
    assert many["march_devices"] == n_dev and many["march_split"] > 0
    assert many["march_opt_out_devices"] == 1 and many["march_opt_out_split"] == 0
