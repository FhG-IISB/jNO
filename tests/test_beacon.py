"""log= on every .solve: live progress from inside compiled loops, and nothing at all without it."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils import beacon


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture
def lines(monkeypatch):
    """Every (name, line) the reader prints, instead of the logger."""
    got = []
    beacon.flush()  # nothing from an earlier test may land here
    monkeypatch.setattr(beacon, "_print", lambda name, line: got.append((name, line)))
    return got


def _loop(n):
    """A fresh function each call: jax.jit caches traces per function, and log= is decided at trace time."""

    def loop(x):
        def body(s):
            k, x = s
            x = 0.5 * x
            if beacon.active():
                beacon.emit("newton", k + 1, jnp.linalg.norm(x), 1e-8, 0.1, step=k + 1, last=k + 1 >= n)
            return k + 1, x

        return jax.lax.while_loop(lambda s: s[0] < n, body, (0, x))[1]

    return loop


def _steps(lines, name="newton"):
    return [int(line.split(":")[0].split()[1]) for n, line in lines if n == name]


def _emitted():
    """How many lines the rings have seen (the newest sequence number of each)."""
    r = beacon._Reader()
    return sum(int(ring["seq"].max()) for ring in (r._ring(False), r._ring(True)) if ring is not None)


def test_without_log_nothing_is_traced():
    jaxpr = str(jax.make_jaxpr(_loop(3))(jnp.ones(4)))
    assert "ffi_call" not in jaxpr and "custom_call" not in jaxpr


def test_every_step_in_order_with_its_values(lines):
    with beacon.session(True):
        jax.block_until_ready(jax.jit(_loop(5))(jnp.ones(4)))
    want = [f"step {k}: residual {2.0 * 0.5**k:.3e} (target 1.0e-08), forcing 1.0e-01" for k in range(1, 6)]
    assert lines == [("newton", w) for w in want]


@pytest.mark.parametrize("log, want", [(3, [3, 6, 9, 10]), (5, [5, 10]), (20, [10]), (True, list(range(1, 11)))])
def test_log_n_reports_every_nth_step_and_the_last(lines, log, want):
    with beacon.session(log):
        jax.block_until_ready(jax.jit(_loop(10))(jnp.ones(2)))
    assert _steps(lines) == want


@pytest.mark.parametrize("bad", [0, -1, 2.5, "10"])
def test_log_must_be_true_false_or_a_positive_whole_number(bad):
    with pytest.raises(ValueError, match="whole number of steps"):
        with beacon.session(bad, "fem.solve"):
            pass


def test_emit_without_log_raises():
    with pytest.raises(RuntimeError, match="guard the call"):
        beacon.emit("newton", 1.0)


def test_a_lapped_reader_says_how_many_lines_it_lost():
    got = []
    beacon._load(beacon._on_gpu())
    r = beacon._Reader(printer=lambda name, line: got.append((name, line)))  # not started: reads only when told
    r.next = {g: int(ring["seq"].max()) + 1 for g in (False, True) if (ring := r._ring(g)) is not None}
    n = 5000  # more than the 4096-slot ring holds between two reads
    with beacon.session(True):
        jax.block_until_ready(jax.jit(_loop(n))(jnp.ones(2)))
    r.poll()
    assert got[0] == ("beacon", f"{n - 4096} lines lost (more than 4096 between two reads)")
    assert len(got) == 1 + 4096 and got[-1][1].startswith(f"step {n}:")


def test_values_are_not_differentiated():
    def f(x):
        if beacon.active():
            beacon.emit("newton", 1.0, jnp.sum(x**2), 0.0, 0.0)
        return jnp.sum(x**3)

    with beacon.session(True):
        g = jax.grad(f)(jnp.arange(3.0))
    np.testing.assert_allclose(g, 3 * np.arange(3.0) ** 2)


def _nonlinear_fem():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.2).domain()
    u, v = d.fem_symbols()
    xi, yi = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    return jno.fem([(1.0 + ui * ui) * (ui.x * vi.x + ui.y * vi.y) - 10.0 * vi, u(xb, yb) - 0.0])


@pytest.mark.parametrize("direct", [True, False])
def test_fem_newton_reports_one_line_per_step_and_the_same_answer(lines, direct):
    fem = _nonlinear_fem()
    kw = {} if direct else {"nonlinear": jno.solve.newton(direct=False)}
    off = np.asarray(fem.solve(**kw))  # compiled and cached without log= first
    assert lines == []
    on = np.asarray(fem.solve(log=True, **kw))
    steps = fem.stats["nonlinear"]["steps"]
    assert _steps(lines) == list(range(1, steps + 1))
    last = float(lines[-1][1].split("residual ")[1].split()[0])
    target = float(lines[-1][1].split("target ")[1].split(")")[0])
    assert last <= target
    np.testing.assert_allclose(on, off, rtol=0, atol=1e-12)  # reporting reads the solve, it does not change it
    # without log= again: the cached program without reporting, the rings stand still
    before = _emitted()
    fem.solve(**kw)
    assert _emitted() == before


def test_a_march_reports_its_time_steps_after_their_newton_steps(lines):
    d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25), time=(0.0, 0.5, 6))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    fem = jno.fem(
        [ui.t * vi + (1.0 + ui * ui) * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0, u(ci[0], ci[1]) - 0.0]
    )
    w_off = np.asarray(fem.solve().fn())  # compiled and cached without log= first
    node = fem.solve(log=True)
    assert lines == []  # a deferred solve reports when it is evaluated
    w = np.asarray(node.fn())
    np.testing.assert_allclose(w, w_off, rtol=0, atol=1e-12)
    times = [line for name, line in lines if name == "time"]
    assert [t.split(",")[0] for t in times] == [f"step {k}: t = {0.1 * k:.6g}" for k in range(1, 6)]
    names = [name for name, _ in lines]
    assert names[0] == "newton" and names[-1] == "time"  # each time step follows its Newton steps
    assert float(times[-1].split("max |u| = ")[1]) == pytest.approx(np.abs(w[-1]).max(), rel=1e-3)


def test_fdm_newton_reports_and_its_compiled_solve_keys_on_log(lines):
    d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).structured())
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    fdm = jno.fdm([-ui.d2(x) - ui.d2(y) + 0.5 * ui**3 - 10.0, u(xb, yb) - 0.0])
    off = np.asarray(fdm.solve())  # compiled and cached without log= first
    on = np.asarray(fdm.solve(log=True))
    assert lines and {n for n, _ in lines} == {"newton"}
    np.testing.assert_allclose(on, off, rtol=0, atol=1e-12)
    before = _emitted()
    fdm.solve()
    assert _emitted() == before


def test_lobpcg_reports_its_iterations(lines):
    d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1))
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    K = jno.fem([ui.x * vi.x + ui.y * vi.y])
    lam, _ = K.eigs(mass=[ui * vi], k=4, precond=jno.precond.jacobi(), tol=1e-6, maxiter=300, log=5)
    assert np.all(np.isfinite(np.asarray(lam)))
    its = _steps(lines, "eigs")
    assert its and all(i % 5 == 0 for i in its[:-1])
    assert float(lines[-1][1].split("residual ")[1].split()[0]) <= 1e-6  # the last line is the converged one


def _training():
    import foundax
    import optax

    dom = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
    x, *_ = dom.variable("interior")
    net = jno.nn.wrap(foundax.mlp(1, output_dim=1, hidden_dims=8, num_layers=2, key=jax.random.PRNGKey(0)))
    net.optimizer(optax.adam(1e-3))
    return jno.core([(net(x) - x).mse])


def _epoch_lines(crux, capsys, **kw):
    capsys.readouterr()
    crux.solve(20, **kw)
    out = capsys.readouterr().out + capsys.readouterr().err
    return [int(line.split("Epoch")[1].split("/")[0]) for line in out.splitlines() if "Epoch" in line and "/20" in line]


def test_core_prints_every_n_epochs_and_the_last(capsys):
    crux = _training()
    assert _epoch_lines(crux, capsys, log=7) == [0, 7, 14, 19]
    assert _epoch_lines(crux, capsys, log=False) == []


def test_a_nonlinear_fem_inside_core_stays_quiet_unless_it_asks(lines, capsys):
    """The case log= per solve exists for: a jno.fem in a jno.core loss reports nothing of its own Newton
    steps, so the training lines stay readable; the same solve with log= reports them, before each epoch line."""
    import optax

    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u, v = d.fem_symbols()
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    k = jno.np.parameter((1,), name="k")
    k.dtype(jnp.float64)
    k.initialize(jax.nn.initializers.constant(1.0))
    k.optimizer(optax.adam(1e-2))
    fem = jno.fem([(k + ui * ui) * (ui.x * vi.x + ui.y * vi.y) - 10.0 * vi, u(xb, yb) - 0.0])
    jno.core([(u * u).integrate(fem)], domain=jno.domain.from_array({"_": np.zeros((1, 1))})).solve(12, log=5)
    assert lines == []
    sol = fem.solve(log=True)
    jno.core([(sol * sol).mean], domain=jno.domain.from_array({"_": np.zeros((1, 1))})).solve(3, log=1)
    assert lines and {n for n, _ in lines} == {"newton"}
