"""The quantities frozen in the RCWA goldens, computed with jNO's own backend (shared by ``regen_goldens.py`` and
``tests/test_rcwa_golden.py``). Everything runs in float64; callers enable x64."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jno._rcwa import fields as jf
from jno._rcwa import sources as jno_sources
from jno._rcwa.stack import StateStack


def build_stack(spec) -> StateStack:
    dtype = jnp.complex128 if spec["dtype"] == "complex128" else jnp.complex64
    layers = [(t, jnp.asarray(m) if isinstance(m, np.ndarray) else m) for t, m in spec["layers"]]
    return StateStack(
        wavelength=spec["wavelength"],
        period=spec["period"],
        amb_sup=spec["amb_sup"],
        amb_sub=spec["amb_sub"],
        layers=layers,
        orders=spec["orders"],
        k_in=spec["k_in"],
        formulation=spec.get("formulation"),
        dtype=dtype,
    )


def plane_wave(spec) -> dict:
    """T, R and the per-order transmitted and reflected efficiencies (with the orders they belong to)."""
    st = build_stack(spec)
    res = st.solve(validate=False)
    pin = float(res.pin)
    return {
        "T": float(res.T),
        "R": float(res.R),
        "orders": np.asarray(st.expansion.basis_coefficients),
        "T_orders": np.asarray(res.flux_trn) / pin,
        "R_orders": -np.asarray(res.flux_rfl) / pin,
    }


def slab_e2(spec, nx: int = 16) -> float:
    """Mean |E|^2 on the mid-film plane of a uniform slab (the incident field has unit amplitude)."""
    st = build_stack(spec)
    planes = jf.field_planes([1.0, float(spec["layers"][0][0]), 1.0], [2, 3, 2], ["sup", 0, "sub"])
    grids = [jnp.asarray([[spec["layers"][0][1]]])]
    ex, ey, ez, _, _, _ = jf.interior_field_amplitudes(
        st, planes, grids, st.wavelength, jnp.asarray(st.k_in), (1.0, 0.0), st.amb_sup, st.amb_sub
    )
    inten, _, _ = jf.volume_intensity(ex, ey, ez, st.expansion, jnp.asarray(st.k_in), st.period, nx)
    return float(np.asarray(inten[..., 3]).mean())  # sup (2 planes), then film z = 0, mid, full


def emitter(spec) -> dict:
    st = build_stack(spec)
    up, down = jno_sources.emission_solve(st, dict(spec["source"]))
    return {"up": float(up), "down": float(down)}


def gradients(spec) -> dict:
    """dT/ds of the contrast scaling ``eps(s) = 1 + (eps - 1)(1 + s)`` of the first film, and dT/dd of its
    thickness, both at the case's design and by autodiff under jit."""
    st = build_stack(spec)
    grid = jnp.asarray(spec["layers"][0][1])
    d0 = float(spec["layers"][0][0])

    def t_of(s, d):
        res = st.solve(eps_grids=[1.0 + (grid - 1.0) * (1.0 + s)], thick=[d], validate=False)
        return jnp.real(res.T)

    ds, dd = jax.jit(jax.grad(t_of, argnums=(0, 1)))(0.0, d0)
    return {"dT_dcontrast": float(ds), "dT_dthickness": float(dd)}
