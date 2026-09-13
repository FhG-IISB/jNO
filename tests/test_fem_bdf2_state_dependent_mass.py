"""BDF2 with a state-dependent mass ``c(u)·u_t``.

``jno.solve.bdf2()`` is a backward-Euler step of size ``2dt/3`` taken from the shifted state
``u* = (4uⁿ - uⁿ⁻¹)/3``. A state-dependent mass is assembled as ``c(u)(u - u_prev)/h`` with ``u_prev`` the
step's own starting state, so the same reduction delivers exactly

    c(uⁿ⁺¹) · (3uⁿ⁺¹ - 4uⁿ + uⁿ⁻¹) / (2dt)

-- BDF2 in its non-conservative form. It used to be refused by name, which left every nonlinear-capacity
problem (variable-density flow, an apparent heat capacity, Richards' equation) at first order in time.

Oracle: a manufactured solution with ``c(u) = 1 + u²``, marched at three step sizes against a fine-``dt``
reference on the SAME mesh, so the spatial error cancels and only the temporal one is measured. The
observed order is 2 for BDF2 and 1 for backward Euler on the identical problem.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

PI = np.pi
T_END = 0.4


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _final_state(n_steps, scheme):
    """March  c(u)·u_t - Δu = f  with c(u) = 1 + u², u* = e^{-t} sin(πx) sin(πy), to T_END."""
    d = jno.shape.rect(0, 0, 1, 1, size=0.15).domain(time=(0.0, T_END, n_steps + 1))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    x0, y0, _ = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    u_star = jno.np.exp(-ti) * jno.np.sin(PI * xi) * jno.np.sin(PI * yi)
    f = (1.0 + u_star * u_star) * (-u_star) + 2.0 * PI**2 * u_star  # c(u*) u*_t - Δu*
    c = 1.0 + ui * ui  # the capacity depends on the unknown: a state-dependent mass
    fem = jno.fem(
        [
            c * ui.t * vi + (ui.x * vi.x + ui.y * vi.y) - f * vi,
            u(xb, yb) - 0.0,
            u(x0, y0) - jno.fn(lambda x, y: jnp.sin(PI * x) * jnp.sin(PI * y), [x0, y0]),
        ]
    )
    sol = fem.solve(time=jno.solve.bdf2()) if scheme == "bdf2" else fem.solve()
    return np.asarray(jno.core([sol.mse]).eval([sol]))[-1].reshape(-1)


@pytest.mark.parametrize("scheme, order", [("bdf2", 2.0), ("backward_euler", 1.0)])
def test_temporal_order_with_a_state_dependent_mass(scheme, order):
    ref = _final_state(256, scheme)
    errs = np.array([np.linalg.norm(_final_state(n, scheme) - ref) for n in (8, 16, 32)])
    rates = np.log2(errs[:-1] / errs[1:])
    assert abs(rates[-1] - order) < 0.25, f"{scheme}: observed temporal orders {np.round(rates, 2)}, expected {order}"


def test_bdf2_beats_backward_euler_at_the_same_step():
    """The point of the change, stated as a number: at 16 steps BDF2's temporal error is well below
    backward Euler's on the same nonlinear-capacity problem."""
    ref_bdf2, ref_be = _final_state(256, "bdf2"), _final_state(256, "backward_euler")
    e_bdf2 = np.linalg.norm(_final_state(16, "bdf2") - ref_bdf2)
    e_be = np.linalg.norm(_final_state(16, "backward_euler") - ref_be)
    assert e_bdf2 < 0.2 * e_be, f"BDF2 error {e_bdf2:.2e} is not clearly below backward Euler's {e_be:.2e}"
