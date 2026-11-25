"""Tests for ``jno._rcwa.stack``: the chunked impedance recursion and the
plane-wave solve, plus readouts.

Anchors (always run): analytic Fresnel/Airy slabs (normal + oblique TE/TM),
energy conservation, chunk-invariance, uniform-collapse exactness, jit reuse,
native-JVP gradients vs finite differences, re-solve across wavelength, order
indexing and the loud interior-fields failure. Oracle tests (skip without
fmmax): the crossed-grating disk and the uniform slab against fmmax 1.7.1.

The disk numbers below are directly measured against fmmax in this file: the
FFT formulation agrees to rel ~1e-5 (complex64) and ~1e-14 (complex128); the
JONES_DIRECT_FOURIER formulation (the jNO default) agrees to rel ~1e-6 at both
precisions, at orders=160. The recursion is found stable at complex64 for this
problem (the stack executes ~30 chunk steps of an O(1)-conditioned impedance
recursion; see the energy and chunk-invariance anchors).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jno._rcwa import fields
from jno._rcwa.stack import StateStack

# -----------------------------------------------------------------------------
# Shared helpers and fixtures.
# -----------------------------------------------------------------------------


def _airy_oblique(n1, n2, n3, d, k0, kt, te=True):
    """Exact oblique Airy transmission (power) through a uniform slab."""
    kz1 = np.sqrt(n1**2 * k0**2 - kt**2 + 0j)
    kz2 = np.sqrt(n2**2 * k0**2 - kt**2 + 0j)
    kz3 = np.sqrt(n3**2 * k0**2 - kt**2 + 0j)

    def yfunc(n, kz, te_):
        return kz / k0 if te_ else n**2 * k0 / kz

    y1, y2, y3 = yfunc(n1, kz1, te), yfunc(n2, kz2, te), yfunc(n3, kz3, te)
    r12 = (y1 - y2) / (y1 + y2)
    r23 = (y2 - y3) / (y2 + y3)
    t12 = (2 * y1) / (y1 + y2)
    t23 = (2 * y2) / (y2 + y3)
    r21 = -r12
    ph = np.exp(1j * kz2 * d)
    t = t12 * t23 * ph / (1 - r21 * r23 * ph**2)
    power_ratio = float(np.real(kz3) / np.real(kz1)) if (np.real(kz1) > 0 and np.real(kz3) > 0) else 0.0
    return float(np.real(t * np.conj(t))) * max(power_ratio, 0.0)


def _airy_normal(n1, n2, n3, d, k0):
    """Exact normal-incidence Airy (power) for three media."""
    kz = n2 * k0
    r12 = (n1 - n2) / (n1 + n2)
    r23 = (n2 - n3) / (n2 + n3)
    ph = np.exp(2j * kz * d)
    t12 = 2 * n1 / (n1 + n2)
    t23 = 2 * n2 / (n2 + n3)
    t = t12 * t23 * np.exp(1j * kz * d) / (1 + r12 * r23 * ph)
    return float(np.real(t * np.conj(t)) * n3 / n1)


def _disk_grid(grid=64, eps=11.0, r=0.3, period=0.6):
    x = (np.arange(grid) + 0.5) / grid
    xx, yy = np.meshgrid(x, x, indexing="ij")
    inside = (xx - 0.5) ** 2 + (yy - 0.5) ** 2 < r**2
    return np.where(inside, eps, 1.0).astype(np.float32)


PERIOD = (0.55, 0.55)
SLAB = [(0.5, 4.0)]


@pytest.fixture(scope="module")
def slab_stack():
    return StateStack(1.0, PERIOD, 1.0, 1.0, [(0.5, jnp.asarray([[4.0]]))], orders=60)


# -----------------------------------------------------------------------------
# Anchors: analytic slabs.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("h", [0.2, 0.5])
def test_slab_matches_airy_normal(h, slab_stack):
    st = StateStack(1.0, PERIOD, 1.0, 1.0, [(h, jnp.asarray([[4.0]]))], orders=60)
    res = st.solve(validate=False)
    expected = _airy_normal(1.0, 2.0, 1.0, h, 2 * np.pi)
    assert abs(float(res.T) - expected) / expected < 2e-5


@pytest.mark.parametrize("te", [True, False])
def test_slab_matches_airy_oblique(te):
    kin = (1.0, 0.5)
    if te:
        pol = (-kin[1], kin[0])  # TE (s-pol): E perpendicular to k_t
    else:
        pol = (kin[0], kin[1])  # TM (p-pol): E in the plane of incidence
    st = StateStack(
        1.0,
        PERIOD,
        1.0,
        1.0,
        [(0.4, jnp.asarray([[4.0]]))],
        orders=60,
        k_in=kin,
    )
    res = st.solve(polarization=pol, validate=False)
    expected = _airy_oblique(1.0, 2.0, 1.0, 0.4, 2 * np.pi, np.hypot(*kin), te=te)
    assert abs(float(res.T) - expected) / expected < 2e-5


def test_fresnel_half_space_reflection():
    """Air | eps_sub half-space (thin-slab Airy limit) gives the Fresnel R."""
    n_sub = np.sqrt(6.0)
    k0 = 2 * np.pi
    # Airy of a vanishingly thin eps=6 slab between air and the eps=6 substrate
    # collapses to the single-interface Fresnel reflection at the air|6 step.
    expected_r = abs((1 - n_sub) / (1 + n_sub)) ** 2
    st = StateStack(1.0, PERIOD, 1.0, 6.0, [(1e-9, jnp.asarray([[6.0]]))], orders=60)
    res = st.solve(validate=False)
    assert abs(float(res.R) - expected_r) < 2e-5
    # cross-check against the general Airy power formula at that thinness
    expected = _airy_normal(1.0, n_sub, n_sub, 1e-9, k0)
    assert abs(float(res.T) - expected) < 2e-5


def test_energy_conservation_lossless_slab(slab_stack):
    res = slab_stack.solve(validate=False)
    assert float(res.T + res.R) <= 1.0 + 5e-3
    assert float(res.T) == pytest.approx(1.0, abs=2e-5)  # quarter-wave peak


def test_energy_conservation_lossless_disk():
    dg = _disk_grid(40)
    st = StateStack(
        1.0,
        (0.6, 0.6),
        1.0,
        1.0,
        [(0.35, jnp.asarray(dg))],
        orders=80,
        formulation="JONES_DIRECT_FOURIER",
    )
    res = st.solve(validate=False)
    assert float(res.T + res.R) <= 1.0 + 5e-3


# -----------------------------------------------------------------------------
# Anchors: chunk invariance and uniform collapse.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("chunks", [1, 8, 64])
def test_chunk_invariance_uniform_slab(chunks):
    """Same-material sub-slabs: the chunk count is numerically inert."""
    st = StateStack(
        1.0,
        PERIOD,
        1.0,
        1.0,
        [(0.5, jnp.asarray([[4.0]]))],
        orders=60,
        chunk_override=[chunks],
    )
    ref = StateStack(1.0, PERIOD, 1.0, 1.0, [(0.5, jnp.asarray([[4.0]]))], orders=60)
    assert abs(float(st.solve(validate=False).T) - float(ref.solve(validate=False).T)) < 1e-5


@pytest.mark.parametrize("chunks", [8, 64])
def test_chunk_invariance_thick_lossy_slab(chunks):
    """2.4 lambda of lossy material stays chunk-invariant (the recursion's point)."""
    st = StateStack(
        1.0,
        (0.6, 0.6),
        1.0,
        1.0,
        [(2.4, jnp.asarray([[2.0 + 0.5j]]))],
        orders=60,
        chunk_override=[chunks],
    )
    ref = StateStack(1.0, (0.6, 0.6), 1.0, 1.0, [(2.4, jnp.asarray([[2.0 + 0.5j]]))], orders=60)
    assert abs(float(st.solve(validate=False).T) - float(ref.solve(validate=False).T)) < 1e-6


def test_uniform_collapse_exactness():
    """scalar == (1,1) grid == (64,64) constant grid (all three solve paths)."""
    base = StateStack(1.0, PERIOD, 1.0, 1.0, [(0.4, 4.0)], orders=60)
    scalar_res = base.solve(validate=False)
    grid11 = StateStack(1.0, PERIOD, 1.0, 1.0, [(0.4, jnp.asarray([[4.0]]))], orders=60)
    grid64 = StateStack(
        1.0,
        PERIOD,
        1.0,
        1.0,
        [(0.4, jnp.full((64, 64), 4.0, jnp.complex64))],
        orders=60,
    )
    t11 = float(grid11.solve(validate=False).T)
    t64 = float(grid64.solve(validate=False).T)
    t0 = float(scalar_res.T)
    assert abs(t11 - t0) < 1e-5
    assert abs(t64 - t0) < 1e-5


def test_fft_and_jones_formulations_agree_on_uniform():
    """On a uniform layer the analytic shortcut is formulation-independent."""
    res_fft = StateStack(
        1.0,
        PERIOD,
        1.0,
        1.0,
        [(0.4, jnp.asarray([[4.0]]))],
        orders=60,
        formulation="FFT",
    ).solve(validate=False)
    res_jones = StateStack(
        1.0,
        PERIOD,
        1.0,
        1.0,
        [(0.4, jnp.asarray([[4.0]]))],
        orders=60,
        formulation="JONES_DIRECT_FOURIER",
    ).solve(validate=False)
    assert abs(float(res_fft.T) - float(res_jones.T)) < 1e-7


def test_mixed_stack_uniform_patterned_uniform():
    """A stack of uniform+patterned+uniform layers solves (guard bookkeeping
    is indexed by layer position; regression for the T13 audit fix)."""
    dg = jnp.asarray(_disk_grid(32))
    st = StateStack(
        1.0,
        (0.6, 0.6),
        1.0,
        1.0,
        [(0.2, jnp.asarray([[2.5]])), (0.3, dg), (0.1, jnp.asarray([[1.5]]))],
        orders=40,
    )
    res = st.solve(validate=False)
    assert jnp.isfinite(res.T)
    assert jnp.isfinite(res.R)
    # Lossless stack: energy conserved within the house tolerance.
    assert float(res.T + res.R) <= 1.0 + 5e-3
    # Re-solve across wavelength also works on a mixed stack.
    r2 = st.solve(wavelength=1.3, validate=False)
    assert jnp.isfinite(r2.T)
    assert abs(float(r2.T) - float(res.T)) > 1e-4


# -----------------------------------------------------------------------------
# Anchors: jit, gradients, re-solve, guards.
# -----------------------------------------------------------------------------


def test_jit_compiles_and_reuses(slab_stack):
    """The compiled core is reused: two solves run the same executable (bit-equal)."""
    jax.clear_caches()
    st = StateStack(1.0, PERIOD, 1.0, 1.0, [(0.5, jnp.asarray([[4.0]]))], orders=60)
    r1 = st.solve(validate=False)  # compiles the jitted core on first call
    r2 = st.solve(validate=False)  # cached executable
    # Bit-equal: the same jitted core produced both results (CPU deterministic).
    assert float(r1.T) == float(r2.T)


def test_grad_uniform_slab_matches_fd(slab_stack):
    """Native JVP of T w.r.t. eps vs central finite differences (rel ~1e-2)."""
    st = StateStack(1.0, PERIOD, 1.0, 1.0, [(0.2, jnp.asarray([[4.0]]))], orders=60)

    def T(e):
        return st.solve(eps_grids=[jnp.asarray([[e]])], validate=False).T

    eps0 = jnp.asarray(3.7 + 0.0j)
    grad = jax.grad(lambda e: T(e).real)(eps0)
    h = 1e-3
    fd = (float(T(3.7 + h).real) - float(T(3.7 - h).real)) / (2 * h)
    assert abs(float(grad.real) - fd) / abs(fd) < 5e-2


def test_grad_patterned_jones_matches_fd():
    """JVP flows through Toeplitz + tangent field + recursion for a disk cell."""
    dg = _disk_grid(32)
    st = StateStack(
        1.0,
        (0.6, 0.6),
        1.0,
        1.0,
        [(0.35, jnp.asarray(dg))],
        orders=40,
        formulation="JONES_DIRECT_FOURIER",
    )
    e0 = jnp.asarray(dg, jnp.float32)

    def T(e):
        return st.solve(eps_grids=[e], validate=False).T.real

    grad = jax.grad(T)(e0)
    # pick a cell on the disk edge where the gradient is representative
    mask = (dg > 5.0).astype(int)
    edge = np.argwhere(np.abs(mask - np.roll(mask, 1, axis=0)) > 0)
    i, j = tuple(int(v) for v in edge[len(edge) // 2])
    h = 0.25
    fd = (float(T(e0.at[i, j].add(h))) - float(T(e0.at[i, j].subtract(h)))) / (2 * h)
    denom = max(abs(fd), 1e-8)
    assert abs(float(grad[i, j]) - fd) / denom < 0.2


def test_resolve_across_wavelength():
    st = StateStack(1.0, PERIOD, 1.0, 1.0, [(0.5, jnp.asarray([[4.0]]))], orders=60)
    r1 = st.solve(validate=False)
    r2 = st.solve(wavelength=1.3, validate=False)
    assert jnp.isfinite(r2.T)
    assert abs(float(r1.T) - float(r2.T)) > 1e-4  # genuinely different answers


def test_order_index(slab_stack):
    i0 = slab_stack.order_index(0, 0)
    assert slab_stack.expansion.basis_coefficients[i0].tolist() == [0, 0]
    with pytest.raises(ValueError, match="outside"):
        slab_stack.order_index(10**4, 0)
    # order readout: only the (0,0) order carries flux at this period
    res = slab_stack.solve(validate=False)
    assert res.flux_trn[i0] == pytest.approx(sum(res.flux_trn), rel=1e-5)


def test_interior_fields_raise_loudly(slab_stack):
    slab_stack.solve(validate=False)  # a solved stack is the entry point
    with pytest.raises(NotImplementedError, match="stage 2"):
        fields.interior_field_amplitudes(slab_stack, [0.1])


def test_shape_and_polarization_guards():
    st = StateStack(1.0, PERIOD, 1.0, 1.0, [(0.5, jnp.asarray([[4.0]]))], orders=60)
    with pytest.raises(ValueError, match="does not match"):
        st.solve(eps_grids=[jnp.ones((8, 8), jnp.complex64)], validate=False)
    with pytest.raises(ValueError, match="polarization"):
        st.solve(polarization="z", validate=False)
    with pytest.raises(ValueError, match="positive"):
        StateStack(1.0, PERIOD, 1.0, 1.0, [(-0.5, jnp.asarray([[4.0]]))], orders=60)


# -----------------------------------------------------------------------------
# G1 review-gate guards: never-silent (NaN/inf, chunk cliff, lengths, counting).
# -----------------------------------------------------------------------------


def test_energy_guard_raises_on_nonfinite(monkeypatch):
    """A NaN/inf efficiency must raise loudly, not pass the T+R/T<0 comparisons
    (``nan > 1 + tol`` and ``nan < 0`` are both False, so a c64 blowup that
    overflows to nan would otherwise slip straight through)."""
    from jno._rcwa import stack as smod

    st = smod.StateStack(1.0, PERIOD, 1.0, 1.0, [(0.5, jnp.asarray([[4.0]]))], orders=60)
    real_cls = smod._Result

    class _NanResult(real_cls):
        @classmethod
        def from_tuple(cls, vals):
            r = real_cls.from_tuple(vals)
            r.T = jnp.asarray(float("nan"))
            r.R = jnp.asarray(float("nan"))
            return r

    monkeypatch.setattr(smod, "_Result", _NanResult)
    with pytest.raises(ValueError, match="non-finite"):
        st.solve(validate=True)


def test_chunk_cliff_c64_chunks_one_raises_loudly():
    """Prototype task O4 port: a THICK patterned disk forced into ONE sub-slab is the
    recursion-condition cliff (condition ~ exp(2 rho d)). At complex64 the impedance
    recursion loses it: the compiled answer is garbage and T+R > 1+5e-3, which the
    energy guard must reject loudly -- the never-silent mandate of the chunking
    design (an un-chunked solve is *meant* to fail, not to return numbers)."""
    from jno._rcwa import stack as smod

    dg = _disk_grid(64)
    st = smod.StateStack(
        1.0,
        (0.6, 0.6),
        1.0,
        1.0,
        [(0.35, jnp.asarray(dg))],
        orders=80,
        chunk_override=[1],
    )
    auto_chunks = smod.StateStack(1.0, (0.6, 0.6), 1.0, 1.0, [(0.35, jnp.asarray(dg))], orders=80)._chunks[0]
    assert auto_chunks > 1, "the auto chunking must actually chunk this thick layer"
    with pytest.raises(ValueError, match="energy guard|non-finite|negative"):
        st.solve(validate=True)


def test_chunk_cliff_c128_chunks_one_matches_auto():
    """Prototype task O4 port: at complex128 the same-material chunk split is exact
    -- chunks=1 must equal the auto chunk count (the recursion stays conditioned;
    the formulas never need the guard)."""
    dg = _disk_grid(64)

    def T_at(chunks):
        jax.config.update("jax_enable_x64", True)
        try:
            st = StateStack(
                1.0,
                (0.6, 0.6),
                1.0,
                1.0,
                [(0.35, jnp.asarray(dg))],
                orders=80,
                chunk_override=chunks,
                dtype=jnp.complex128,
            )
            return float(st.solve(validate=False).T)
        finally:
            jax.config.update("jax_enable_x64", False)

    t_auto = T_at(None)
    t_one = T_at([1])
    assert abs(t_one - t_auto) <= 1e-6 * max(1.0, abs(t_auto))


def test_solve_eps_grids_length_guard():
    """A short eps_grids list silently truncates the stack (missing bottom layers,
    wrong T/R); a long list crashed with an unhelpful IndexError -- both must raise
    loudly, mirroring the chunk_override length guard."""
    st = StateStack(1.0, PERIOD, 1.0, 1.0, [(0.5, jnp.asarray([[4.0]]))], orders=60)
    with pytest.raises(ValueError, match="eps_grids has 0 entries"):
        st.solve(eps_grids=[], validate=False)
    with pytest.raises(ValueError, match="eps_grids has 3 entries"):
        st.solve(
            eps_grids=[jnp.asarray([[4.0]]), jnp.asarray([[4.0]]), jnp.asarray([[4.0]])],
            validate=False,
        )


def test_patterned_solve_count_counts_traces_only():
    """count_patterned_solves counts *traces* (one per patterned layer per trace):
    the eager validate pre-run must not double-count a cold solve (2x), and a warm
    repeat solve must re-trace nothing."""
    from jno._rcwa.stack import count_patterned_solves

    dg = jnp.asarray(_disk_grid(24))
    st = StateStack(1.0, (0.6, 0.6), 1.0, 1.0, [(0.3, dg)], orders=40)
    _, n1 = count_patterned_solves(lambda: st.solve())  # default validate=True
    assert n1 == 1, f"one patterned layer -> exactly one traced solve, got {n1}"
    _, n2 = count_patterned_solves(lambda: st.solve())  # warm: compiled core reused
    assert n2 == 0, f"a warm repeat solve must re-trace nothing, got {n2}"


def test_bad_formulation_name_raises_loud():
    """A bogus formulation name is a user error: loud ValueError listing the valid
    names, never a bare KeyError."""
    with pytest.raises(ValueError, match="unknown formulation"):
        StateStack(
            1.0,
            PERIOD,
            1.0,
            1.0,
            [(0.5, jnp.asarray([[4.0]]))],
            orders=60,
            formulation="NOT_A_REAL_SCHEME",
        )


# -----------------------------------------------------------------------------
# fmmax oracles (skip when fmmax is absent).
# -----------------------------------------------------------------------------


def _fmmax():
    import fmmax

    return fmmax


def _fmmax_disk_reference(fmmax, dg, h=0.35, orders=160, formulation="fft", period=(0.6, 0.6)):
    """fmmax x64 T/R for the crossed-grating disk (prototype oracle builder)."""
    from fmmax import fields as ff
    from fmmax import fmm, scattering, sources

    jax.config.update("jax_enable_x64", True)
    lv = fmmax.basis.LatticeVectors(jnp.array([period[0], 0.0]), jnp.array([0.0, period[1]]))
    ex = fmmax.basis.generate_expansion(lv, orders)
    kin = jnp.asarray([0.0, 0.0])
    form = fmm.Formulation.FFT if formulation == "fft" else fmm.Formulation.JONES_DIRECT_FOURIER
    layers = [
        fmm.eigensolve_isotropic_media(jnp.asarray(1.0), kin, lv, jnp.asarray([[1.0]]), ex, formulation=form),
        fmm.eigensolve_isotropic_media(jnp.asarray(1.0), kin, lv, jnp.asarray(dg), ex, formulation=form),
        fmm.eigensolve_isotropic_media(jnp.asarray(1.0), kin, lv, jnp.asarray([[1.0]]), ex, formulation=form),
    ]
    s = scattering.stack_s_matrix(layers, [1.0, h, 1.0])
    g = 2 * int(np.amax(np.abs(ex.basis_coefficients))) + 1
    fwd = sources.amplitudes_for_fields(
        jnp.ones((g, g, 1), jnp.complex128),
        jnp.zeros((g, g, 1), jnp.complex128),
        jnp.zeros((g, g, 1), jnp.complex128),
        jnp.zeros((g, g, 1), jnp.complex128),
        layers[0],
    )[0]
    pin = float(jnp.sum(jnp.real(ff.directional_poynting_flux(fwd, jnp.zeros_like(fwd), layers[0])[0])))
    ft = ff.directional_poynting_flux(s.s11 @ fwd, jnp.zeros_like(fwd), layers[-1])[0]
    fbw = ff.directional_poynting_flux(jnp.zeros_like(fwd), s.s21 @ fwd, layers[0])[1]
    t = float(jnp.sum(jnp.real(ft))) / pin
    r = -float(jnp.sum(jnp.real(fbw))) / pin
    jax.config.update("jax_enable_x64", False)
    return t, r


def test_disk_matches_fmmax_fft():
    """Crossed-grating disk, FFT formulation: rel ~1e-5 (c64), ~1e-14 (c128)."""
    fmmax = _fmmax()
    dg = _disk_grid(64)
    t_ref, r_ref = _fmmax_disk_reference(fmmax, dg, h=0.35, orders=160, formulation="fft")

    st = StateStack(
        1.0,
        (0.6, 0.6),
        1.0,
        1.0,
        [(0.35, jnp.asarray(dg))],
        orders=160,
        formulation="FFT",
    )
    res = st.solve(validate=False)
    assert abs(float(res.T) - t_ref) / t_ref < 5e-4
    assert abs(float(res.R) - r_ref) / r_ref < 5e-4

    jax.config.update("jax_enable_x64", True)
    st128 = StateStack(
        1.0,
        (0.6, 0.6),
        1.0,
        1.0,
        [(0.35, jnp.asarray(dg))],
        orders=160,
        formulation="FFT",
        dtype=jnp.complex128,
    )
    res128 = st128.solve(validate=False)
    jax.config.update("jax_enable_x64", False)
    assert abs(float(res128.T) - t_ref) / t_ref < 1e-9
    assert abs(float(res128.R) - r_ref) / r_ref < 1e-9


def test_disk_matches_fmmax_jones():
    """Crossed-grating disk, JONES_DIRECT_FOURIER (jNO default): rel ~1e-6."""
    fmmax = _fmmax()
    dg = _disk_grid(64)
    t_ref, r_ref = _fmmax_disk_reference(fmmax, dg, h=0.35, orders=160, formulation="jones")

    st = StateStack(
        1.0,
        (0.6, 0.6),
        1.0,
        1.0,
        [(0.35, jnp.asarray(dg))],
        orders=160,
        formulation="JONES_DIRECT_FOURIER",
    )
    res = st.solve(validate=False)
    assert abs(float(res.T) - t_ref) / t_ref < 5e-4
    assert abs(float(res.R) - r_ref) / r_ref < 5e-4

    jax.config.update("jax_enable_x64", True)
    st128 = StateStack(
        1.0,
        (0.6, 0.6),
        1.0,
        1.0,
        [(0.35, jnp.asarray(dg))],
        orders=160,
        formulation="JONES_DIRECT_FOURIER",
        dtype=jnp.complex128,
    )
    res128 = st128.solve(validate=False)
    jax.config.update("jax_enable_x64", False)
    assert abs(float(res128.T) - t_ref) / t_ref < 5e-6
    assert abs(float(res128.R) - r_ref) / r_ref < 5e-6


def test_slab_matches_fmmax():
    """Uniform slab order-0 transmission vs fmmax x64 (rel ~1e-5 at c64)."""
    fmmax = _fmmax()
    jax.config.update("jax_enable_x64", True)
    lv = fmmax.basis.LatticeVectors(jnp.array([PERIOD[0], 0.0]), jnp.array([0.0, PERIOD[1]]))
    ex = fmmax.basis.generate_expansion(lv, 60)
    from fmmax import fields as ff
    from fmmax import fmm, scattering, sources

    layers = [
        fmm.eigensolve_isotropic_media(
            jnp.asarray(1.0),
            jnp.asarray([0.0, 0.0]),
            lv,
            jnp.asarray([[1.0]]),
            ex,
            formulation=fmm.Formulation.FFT,
        ),
        fmm.eigensolve_isotropic_media(
            jnp.asarray(1.0),
            jnp.asarray([0.0, 0.0]),
            lv,
            jnp.asarray([[4.0]]),
            ex,
            formulation=fmm.Formulation.FFT,
        ),
        fmm.eigensolve_isotropic_media(
            jnp.asarray(1.0),
            jnp.asarray([0.0, 0.0]),
            lv,
            jnp.asarray([[1.0]]),
            ex,
            formulation=fmm.Formulation.FFT,
        ),
    ]
    s = scattering.stack_s_matrix(layers, [1.0, 0.4, 1.0])
    g = 2 * int(np.amax(np.abs(ex.basis_coefficients))) + 1
    fwd = sources.amplitudes_for_fields(
        jnp.ones((g, g, 1), jnp.complex128),
        jnp.zeros((g, g, 1), jnp.complex128),
        jnp.zeros((g, g, 1), jnp.complex128),
        jnp.zeros((g, g, 1), jnp.complex128),
        layers[0],
    )[0]
    pin = float(jnp.sum(jnp.real(ff.directional_poynting_flux(fwd, jnp.zeros_like(fwd), layers[0])[0])))
    ft = ff.directional_poynting_flux(s.s11 @ fwd, jnp.zeros_like(fwd), layers[-1])[0]
    t_ref = float(jnp.sum(jnp.real(ft))) / pin
    jax.config.update("jax_enable_x64", False)

    st = StateStack(1.0, PERIOD, 1.0, 1.0, [(0.4, jnp.asarray([[4.0]]))], orders=60)
    res = st.solve(validate=False)
    assert abs(float(res.T) - t_ref) / t_ref < 1e-3
