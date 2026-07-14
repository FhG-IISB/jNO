"""Rigorous Coupled-Wave Analysis (RCWA / Fourier Modal Method) for periodic, layered structures.

RCWA is semi-analytic in the propagation direction: each layer is expanded in a truncated Fourier
basis in-plane and solved by an eigenmode decomposition, then layers are stitched with a scattering
matrix. For **periodic, piecewise-z-invariant** structures — extruded metasurface pillars are the
canonical case — this is far cheaper than a full 3-D volumetric solve.

Two entry points, both on `jno.rcwa`:

* **From a jNO problem (the front door).** Hand it the same constraint list you would give
  :func:`jno.fem` (or the already-built ``FEM``) — nothing else is required::

      rc  = jno.rcwa(constraints, orders=300)
      sol = rc.solve()                     # everything inferred from the traced problem
      rc.spec                              # the inferred RcwaSpec (available without fmmax)

  Inferred from the problem: **periodicity + period** (from the Floquet ties — absent ⇒ raise, never
  assumed), the **super/substrate ambients** (the z-normal radiation faces), the **permittivity**
  (the ``K0**2*eps`` coefficient recovered from the scalar Helmholtz volume term, sampled along z and
  grouped by :func:`detect_layers`), the **wavelength** (``k0`` from the vacuum superstrate, or pass
  ``wavelength=`` to override), and the **incident wave** (illuminated face + transverse angle
  ``k_in``, read from the assembled forcing).

* **Explicit layers (the backend).** :class:`Rcwa` takes a hand-built ``[(thickness, eps), ...]`` stack;
  this is what the front door constructs internally.

Built on `fmmax` (a differentiable JAX Fourier Modal Method), imported lazily so the core install stays
lean. Enable with the ``rcwa`` extra::

    pip install jax-neural-operators[rcwa]      # or: pixi run -e rcwa ...

It **never fails silently**: every inference (periodicity, layer invariance, order/grid Nyquist,
wavelength, energy balance, a recognisable source) is validated and raises :class:`RcwaError` with a
concrete fix. Two `fmmax` conventions are baked in: the Poynting flux is summed *with sign* (an
``abs()`` sum over-counts and yields ``T+R>1``), and ``JONES_DIRECT_FOURIER`` is the default
factorization (the naive rule converges poorly for high-contrast dielectrics such as a-Si).

References:
 * M. G. Moharam & T. K. Gaylord, "Rigorous coupled-wave analysis of planar-grating diffraction",
   J. Opt. Soc. Am. 71, 811 (1981).
 * M. G. Moharam, D. A. Pommet, E. B. Grann & T. K. Gaylord, "Stable implementation of the enhanced
   transmittance matrix approach", J. Opt. Soc. Am. A 12, 1077 (1995).
 * fmmax: https://github.com/facebookresearch/fmmax
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np


def _concrete(x):
    """True when ``x`` is a real (non-traced) value, so a never-silent guard may inspect it. Under
    jit/grad the value is a tracer -- the guard steps aside (validation ran on the eager forward pass)."""
    return not isinstance(x, jax.core.Tracer)


_FMMAX_HINT = (
    "jno.rcwa needs the optional fmmax backend: pip install jax-neural-operators[rcwa] "
    "(or `pixi run -e rcwa ...`, or `pip install fmmax`)."
)


def _fmmax():
    try:
        import fmmax

        return fmmax
    except ImportError as e:  # never a silent no-op
        raise ImportError(_FMMAX_HINT) from e


class RcwaError(ValueError):
    """Raised for any ill-posed RCWA problem — always loud, never a silent wrong answer."""


# =====================================================================================
# Layer auto-detection: group z-invariant slabs of a sampled permittivity into RCWA layers.
# =====================================================================================
def detect_layers(E, z, tol=1e-3, slices=None):
    """Group a z-sampled permittivity into RCWA layers.

    Parameters
    ----------
    E:
        ``(Nz, Ng, Ng)`` permittivity sampled on the unit-cell grid at each height ``z``.
    z:
        ``(Nz,)`` ascending heights.
    tol:
        In-plane change below which two adjacent z-slices are the same material.
    slices:
        If given, staircase a continuously-varying ``eps`` into this many layers instead of raising.

    Returns
    -------
    list[tuple[float, np.ndarray]]
        ``[(thickness, eps_xy), ...]`` super→substrate; the two ambients have thickness ``inf``.

    Raises
    ------
    RcwaError
        If ``eps`` varies continuously in z and ``slices`` is None (RCWA needs invariant slabs).
    """
    Nz = len(z)
    if E.shape[0] != Nz:
        raise RcwaError(f"detect_layers: E has {E.shape[0]} z-slices but z has {Nz}.")
    step = np.array([np.max(np.abs(E[k] - E[k - 1])) for k in range(1, Nz)])
    boundary = step > tol
    edges = [0] + [k for k in range(1, Nz) if boundary[k - 1]] + [Nz]
    slabs = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    thin = sum((b - a) <= 2 for a, b in slabs)
    if slices is None and len(slabs) > 4 and thin > 0.3 * len(slabs):
        raise RcwaError(
            f"eps varies continuously in z ({thin}/{len(slabs)} slabs are 1-cell thin) -> not "
            f"layer-invariant. Pass slices=N to staircase it, or the geometry isn't RCWA-appropriate."
        )
    if slices is not None:
        idx = np.linspace(0, Nz - 1, slices + 1).round().astype(int)
        slabs = [(idx[i], idx[i + 1]) for i in range(len(idx) - 1)]

    layers, report, zmids = [], [], []
    for i, (a, b) in enumerate(slabs):
        mid = (a + b) // 2
        var = float(np.max(np.abs(E[a:b] - E[mid])))
        if var > tol and slices is None:
            raise RcwaError(
                f"slab z=[{z[a]:.3f},{z[min(b, Nz - 1)]:.3f}] is not z-invariant (var={var:.2g}); pass slices=."
            )
        thick = np.inf if (i == 0 or i == len(slabs) - 1) else float(z[b - 1] - z[a])
        eps_xy = E[mid]
        kind = "uniform" if np.ptp(eps_xy) < tol else "patterned"
        layers.append((thick, eps_xy))
        zmids.append(float(z[mid]))
        report.append(
            f"  layer {i}: z=[{z[a]:.3f},{z[min(b - 1, Nz - 1)]:.3f}] {kind} eps~[{eps_xy.min():.2f},{eps_xy.max():.2f}]"
        )
    detect_layers.last_report = "detected layers:\n" + "\n".join(report)
    detect_layers.last_zmid = zmids  # representative z of each layer -- lets a param sweep re-sample eps
    return layers


# =====================================================================================
# The forward engine (explicit layers) — the fmmax backend the front door constructs.
# =====================================================================================
class _Sol:
    def __init__(self, fm, s, layers, expansion, nt, Pin, wavelength, thick=None, period=None):
        self._fm, self._s, self._layers, self._ex = fm, s, layers, expansion
        self._nt, self._Pin, self._wl = nt, Pin, wavelength
        self._thick, self._period = thick, period
        self._fwd = np.zeros((2 * nt, 1), complex)

    def _flux(self, amps, layer, backward=False):
        zero = jnp.zeros_like(amps)
        f, b = self._fm.directional_poynting_flux(zero if backward else amps, amps if backward else zero, layer)
        return jnp.reshape(b if backward else f, (-1,))  # SIGNED — energy-correct; jax-native (differentiable)

    def efficiency(self, kind):
        """Total transmitted (``"T"``) or reflected (``"R"``) power fraction over propagating orders.

        Returns a JAX scalar -- differentiable in the design (permittivity) so ``jax.grad`` of a
        transmission/reflection objective flows through the modal solve."""
        if kind == "T":
            return jnp.sum(self._flux(self._s.s11 @ self._fwd, self._layers[-1])) / self._Pin
        if kind == "R":
            return -jnp.sum(self._flux(self._s.s21 @ self._fwd, self._layers[0], backward=True)) / self._Pin
        raise RcwaError(f"efficiency(kind): kind must be 'T' or 'R', got {kind!r}")

    def order(self, m, n):
        """Diffraction efficiency into transmitted order ``(m, n)`` (raises if outside the truncation).
        Returns a differentiable JAX scalar."""
        bc = np.asarray(self._ex.basis_coefficients)  # static (truncation geometry) -> plain numpy index
        hit = np.where((bc[:, 0] == m) & (bc[:, 1] == n))[0]
        if len(hit) == 0:
            raise RcwaError(f"order ({m},{n}) is outside the truncation ({self._nt} terms); raise orders=.")
        i = int(hit[0])
        tf = jnp.abs(
            jnp.reshape(
                self._fm.directional_poynting_flux(
                    self._s.s11 @ self._fwd, jnp.zeros((2 * self._nt, 1), complex), self._layers[-1]
                )[0],
                (-1,),
            )
        )
        return (tf[i] + tf[i + self._nt]) / self._Pin

    def field(self, y_frac=0.5, nx=80, density=40.0):
        """Reconstruct the real-space electric field on a vertical (x–z) slice at ``y = y_frac·Py``.

        Returns ``(intensity, extent, layer_z)``: ``intensity`` = ``|E|²`` of shape ``(nz, nx)`` (z
        vertical, x horizontal) for ``imshow``; ``extent`` = ``[0, Px, z_min, z_max]``; ``layer_z`` = the
        z-interfaces between layers (to annotate the patterned slab). Needs the finite ambient
        thickness recorded at solve time (semi-infinite ambients are shown as unit-thick slabs)."""
        fm = self._fm
        if self._thick is None:
            raise RcwaError("field() needs the layer thicknesses; call .solve() (which records them).")
        znum = [max(4, int(round(float(t) * density))) for t in self._thick]
        smi = fm.stack_s_matrices_interior(self._layers, self._thick)
        amps = fm.stack_amplitudes_interior(smi, self._fwd, np.zeros_like(self._fwd))
        efield, _h, (x, y, z) = fm.stack_fields_3d(amps, self._layers, self._thick, znum, grid_shape=(nx, nx))
        E = np.asarray(efield)  # (3, nx, ny, nz, 1)
        inten = np.sum(np.abs(E) ** 2, axis=0)[..., 0]  # (nx, ny, nz)
        yv = np.asarray(y)
        j = int(np.argmin(np.abs(yv[0, :] - y_frac * self._period[1])))
        slab = inten[:, j, :].T  # (nz, nx): z vertical, x horizontal
        zc = np.asarray(z).reshape(-1)
        layer_z = list(np.cumsum([float(t) for t in self._thick]))[:-1]
        return slab, [0.0, float(self._period[0]), float(zc.min()), float(zc.max())], layer_z


class Rcwa:
    """A periodic layered RCWA problem with an **explicit** layer stack (the backend engine).

    Most users reach this through :func:`jno.rcwa` with a problem list; construct it directly only when
    you already have the ``[(thickness, eps), ...]`` stack, period and truncation in hand.
    """

    def __init__(
        self,
        layers,
        *,
        period,
        orders,
        wavelength=None,
        k_in=(0.0, 0.0),
        formulation="JONES_DIRECT_FOURIER",
        assume_periodic=False,
    ):
        fm = _fmmax()
        if period is None or len(period) != 2 or period[0] <= 0 or period[1] <= 0:
            raise RcwaError(f"period must be (Px,Py) > 0, got {period!r}.")
        if not assume_periodic:
            raise RcwaError(
                "RCWA is periodic in-plane. If your cell is genuinely a period/supercell, pass "
                "assume_periodic=True. (The jno.rcwa front door sets this only after confirming the "
                "problem has Floquet ties.)"
            )
        if not layers:
            raise RcwaError("layers is empty: need at least [superstrate, ..., substrate].")
        for k, (t, e) in enumerate(layers):
            if not (t is None or (np.isreal(t) and t > 0) or t == np.inf):
                raise RcwaError(f"layer {k} thickness must be > 0 (or inf for ambient), got {t!r}.")
            eg = np.asarray(e)
            if eg.ndim == 2 and (eg.shape[0] < 4 or eg.shape[1] < 4):
                raise RcwaError(f"layer {k} eps grid {eg.shape} is too coarse to rasterize; sample it finer.")
        grids = [np.asarray(e).shape[0] for _, e in layers if np.asarray(e).ndim == 2]
        if grids:
            ng = min(grids)
            per_axis = int(np.ceil(np.sqrt(orders)))
            if ng < 2 * per_axis:
                raise RcwaError(
                    f"eps grid {ng}x{ng} under-resolves orders={orders} (~{per_axis}/axis); "
                    f"need >= {2 * per_axis} grid points/axis or fewer orders."
                )
        self.fm, self.layers_spec, self.period = fm, layers, period
        self.orders, self.wavelength, self.k_in = orders, wavelength, tuple(k_in)
        self.formulation = fm.Formulation[formulation]
        self.assume_periodic = True

    def solve(self, inc=None, wavelength=None, k_in=None, layers=None):
        """Solve the stack and return a :class:`_Sol`. Raises if the wavelength is unknown or energy is
        not conserved.

        ``layers`` optionally overrides the construction layer stack with ``[(thickness, eps), ...]`` at
        solve time -- pass JAX permittivity grids here to differentiate the solve in the design (the
        construction-time shape guards run once, eagerly, so the solve itself stays trace-clean)."""
        fm = self.fm
        wl = wavelength if wavelength is not None else self.wavelength
        if wl is None:
            raise RcwaError(
                "wavelength is unknown: pass wavelength= to solve() or at construction; it sets every "
                "layer's eigenmodes and is never defaulted."
            )
        layers_spec = self.layers_spec if layers is None else layers
        kin = np.asarray(self.k_in if k_in is None else k_in, float)
        lv = fm.LatticeVectors(u=np.array([self.period[0], 0.0]), v=np.array([0.0, self.period[1]]))
        ex = fm.generate_expansion(lv, approximate_num_terms=self.orders)
        nt = ex.num_terms

        def solve_layer(e):
            eg = jnp.asarray(e) + 0j  # jax-native: a jax permittivity grid flows -> differentiable in eps
            if eg.ndim == 0:
                eg = jnp.full((1, 1), eg)
            return fm.eigensolve_isotropic_media(jnp.asarray(wl), kin, lv, eg, ex, formulation=self.formulation)

        layers = [solve_layer(e) for _, e in layers_spec]
        thick = [jnp.asarray(1.0 if t is None or t == np.inf else t) for t, _ in layers_spec]
        s = fm.stack_s_matrix(layers, thick)
        # Incidence: excite the genuinely forward-propagating mode in the superstrate and normalise by
        # ITS flux. fmmax sorts eigenmodes by eigenvalue, so index 0 need not be the 0th forward order,
        # and a one-hot forward amplitude can carry zero forward flux -- pick the max-positive-flux mode.
        # (The superstrate is design-independent, so this is effectively constant, but stays jax-native so
        # the whole solve traces under grad/jit.)
        flux_sup = jnp.real(jnp.reshape(fm.eigenmode_poynting_flux(layers[0]), (-1,)))
        idx = jnp.argmax(flux_sup)
        Pin = flux_sup[idx]
        if _concrete(Pin) and float(Pin) <= 0:
            raise RcwaError("no forward-propagating incident mode in the superstrate; check wavelength/period.")
        fwd = jnp.zeros((2 * nt, 1), complex).at[idx, 0].set(1.0)
        sol = _Sol(fm, s, layers, ex, nt, Pin, wl, thick=thick, period=self.period)
        sol._fwd = fwd
        T, R = sol.efficiency("T"), sol.efficiency("R")
        if _concrete(T):  # never-silent runtime guards run on the eager forward pass; they step aside under trace
            if not (np.isfinite(float(T)) and np.isfinite(float(R))):
                raise RcwaError(
                    "non-finite efficiency (NaN/inf): likely a Rayleigh/Wood anomaly where a diffraction "
                    "order is exactly grazing (the wavelength coincides with a period resonance, e.g. "
                    f"wavelength == period). Nudge the wavelength or period. (period={self.period}, wl={wl})"
                )
            if float(T) + float(R) > 1.0 + 5e-3:
                raise RcwaError(
                    f"energy not conserved: T+R={float(T) + float(R):.4f} > 1 at orders={self.orders}. The modal "
                    f"solve is not converged (raise orders) or the structure is unphysical."
                )
        return sol


# =====================================================================================
# Front door: infer an RCWA problem from a jNO constraint list / FEM problem + eps.
# =====================================================================================
@dataclass
class RcwaSpec:
    """Everything the RCWA engine needs, inferred from a jNO problem. Inspectable without fmmax."""

    period: tuple
    layers: list  # [(thickness, eps_xy), ...]
    wavelength: float
    k_in: tuple = (0.0, 0.0)
    source_face: str = ""
    ambient_faces: tuple = ()
    periodic_axes: dict = field(default_factory=dict)

    def __repr__(self):
        return (
            f"RcwaSpec(period={tuple(round(p, 4) for p in self.period)}, layers={len(self.layers)}, "
            f"wavelength={self.wavelength}, k_in={tuple(round(k, 4) for k in self.k_in)}, "
            f"source_face={self.source_face!r})"
        )


def _walk_nodes(node):
    from jno._fem import _walk

    return _walk(node)


def _model_calls(node):
    return [nd for nd in _walk_nodes(node) if type(nd).__name__ == "ModelCall"]


def _as_fem(problem):
    """Accept a constraint list or a built FEM; return the FEM object."""
    from jno._fem import FEM
    from jno._fem import fem as _fem

    if isinstance(problem, FEM):
        return problem
    if isinstance(problem, (list, tuple)):
        return _fem(list(problem))
    raise RcwaError(
        "jno.rcwa(problem, ...): problem must be the constraint list you'd pass to jno.fem, or a built "
        f"FEM object; got {type(problem).__name__}."
    )


def _region_centroid(domain, tag):
    br = getattr(domain, "_boundary_regions", {}).get(tag)
    if br is None or len(getattr(br, "points", [])) == 0:
        raise RcwaError(f"boundary region {tag!r} has no points; cannot locate it.")
    return np.asarray(br.points).mean(0)


def _periodic_period(femobj, domain):
    """Recover {axis: (tagA, tagB)} and (Px, Py) from the Floquet ties. Raise if not periodic in x and y."""
    from jno._fem import _periodic_tie_spec

    ties = []
    for c in getattr(femobj, "_constraints", []):
        spec = _periodic_tie_spec(c, domain)
        if spec is not None:
            ties.append((spec[0], spec[1]))  # (master, slave) tags
    if not ties:
        raise RcwaError(
            "no Floquet/periodic ties found: RCWA solves an infinitely periodic cell. Author the "
            "in-plane boundaries as periodic ties (u(left)-u(right), u(front)-u(back)); a finite "
            "aperture with absorbing side walls is NOT periodic and cannot be solved by RCWA."
        )
    axes, period = {}, {}
    for a, b in ties:
        ca, cb = _region_centroid(domain, a), _region_centroid(domain, b)
        d = cb - ca
        ax = int(np.argmax(np.abs(d)))  # dominant separation axis: 0=x, 1=y
        axes["xyz"[ax]] = (a, b)
        period["xyz"[ax]] = float(abs(d[ax]))
    if "x" not in period or "y" not in period:
        raise RcwaError(
            f"RCWA needs periodicity in BOTH x and y; found ties only along {sorted(period)}. Add the missing periodic tie."
        )
    return axes, (period["x"], period["y"])


def _z_ambient_faces(domain):
    """The two z-normal radiation faces (bottom, top), by z-extent of the boundary-region centroids."""
    pts = np.asarray(domain.points)
    zmin, zmax = float(pts[:, 2].min()), float(pts[:, 2].max())
    normals = getattr(domain, "normals_by_tag", {})
    bottom = top = None
    for tag, br in getattr(domain, "_boundary_regions", {}).items():
        if tag == "boundary" or br is None or len(getattr(br, "points", [])) == 0:
            continue
        nrm = np.asarray(normals.get(tag, np.zeros((1, 3)))).mean(0)
        if abs(nrm[2]) < 0.8 * (np.linalg.norm(nrm) + 1e-30):
            continue  # not a z-normal face
        cz = np.asarray(br.points)[:, 2].mean()
        if abs(cz - zmin) < abs(cz - zmax):
            bottom = tag
        else:
            top = tag
    if bottom is None or top is None:
        raise RcwaError(
            "could not identify the two z-normal ambient faces (superstrate/substrate). Tag the top and "
            "bottom faces of the cell so RCWA can place the ambients."
        )
    return bottom, top


def _param_values(femobj, eps):
    """{parameter_name: current value} for every trainable parameter the operator references. Uses the
    parameter's *initialized* value (not zeros) so a parameter that is a physical constant -- e.g. K0 in
    the source term -- doesn't vanish when the operator is assembled to read the source."""
    out = {}
    nodes = list(_model_calls(eps))
    for c in getattr(femobj, "_constraints", []):
        nodes += _model_calls(c)
    for mc in nodes:
        m = getattr(mc, "model", None)
        name = getattr(m, "_parameter_name", None)
        if name is None:
            continue
        out[name] = jnp.asarray(m.module.value)
    return out


def _eval_expr_nodes(node, domain):
    """Evaluate a coefficient expression at every mesh node (nodal parameters use their current value)."""
    import jax.numpy as jnp

    from jno.trace import Variable
    from jno.trace_evaluator import TraceEvaluator

    node_coords = jnp.asarray(np.asarray(domain.points))
    table = {mc.model.layer_id: mc.model.module for mc in _model_calls(node)}
    tags = {v.tag for v in _walk_nodes(node) if isinstance(v, Variable)}
    ctx = {t: node_coords for t in tags} if tags else {}
    return np.asarray(TraceEvaluator(table).evaluate(node, context=ctx)).reshape(-1)


# --- recover the permittivity coefficient (K0^2 * eps) from the volume weak form ------------------
def _add_split(node, sign=1, out=None):
    """Flatten a tree of +/- into signed additive summands."""
    from jno.trace import BinaryOp

    if out is None:
        out = []
    if isinstance(node, BinaryOp) and node.op in ("+", "-"):
        _add_split(node.left, sign, out)
        _add_split(node.right, sign if node.op == "+" else -sign, out)
    else:
        out.append((sign, node))
    return out


def _mul_factors(node, out=None):
    """Flatten a product tree into its multiplicative factors."""
    from jno.trace import BinaryOp

    if out is None:
        out = []
    if isinstance(node, BinaryOp) and node.op == "*":
        _mul_factors(node.left, out)
        _mul_factors(node.right, out)
    else:
        out.append(node)
    return out


def _volume_constraint(femobj):
    cls = list(getattr(femobj, "classification", []))
    cons = list(getattr(femobj, "_constraints", []))
    vols = [c for c, k in zip(cons, cls) if k == "volume"]
    if len(vols) != 1:
        raise RcwaError(
            f"expected exactly one volume weak-form term, found {len(vols)} (classification={cls}); "
            "RCWA infers the permittivity from a single scalar Helmholtz volume term."
        )
    return vols[0]


def _extract_permittivity_coeff(volume_term):
    """Pull the value-channel (mass) coefficient ``K0**2 * eps`` out of a scalar Helmholtz volume term.

    A Helmholtz volume form is ``grad(u).grad(v) - K0**2 * eps * (u*v)``: the stiffness summands carry
    trial/test inside ``Jacobian`` nodes, while the mass summand carries them as bare TrialFunction /
    TestFunction values. We split additively, find the summand with bare trial x test, and drop those
    two factors — what remains is ``K0**2 * eps``.
    """
    import functools
    import operator

    from jno.trace import TestFunction, TrialFunction

    expr = getattr(volume_term, "expr", volume_term)
    mass = []
    for _sign, summand in _add_split(expr):
        fac = _mul_factors(summand)
        tv = [f for f in fac if isinstance(f, TrialFunction)]
        te = [f for f in fac if isinstance(f, TestFunction)]
        if not (tv and te):
            continue  # stiffness / other channel
        for f in tv + te:
            if getattr(f, "value_shape", ()) != ():
                raise RcwaError(
                    "RCWA infers permittivity from a SCALAR Helmholtz term, but this field is "
                    f"vector/tensor (value_shape={f.value_shape}). Scalar problems only for now."
                )
        coeff_factors = [f for f in fac if not isinstance(f, (TrialFunction, TestFunction))]
        if not coeff_factors:
            raise RcwaError("mass term has no coefficient factor; cannot recover permittivity.")
        mass.append(functools.reduce(operator.mul, coeff_factors))
    if not mass:
        raise RcwaError(
            "could not find a K0^2*eps*(u*v) mass term in the volume weak form; RCWA needs a scalar "
            "Helmholtz term. If the permittivity is authored unusually, isolate it as a named coefficient."
        )
    return functools.reduce(operator.add, mass)


def _has_nodal_field(node):
    """True if the coefficient depends on a per-node field parameter (needs mesh interpolation)."""
    return any(getattr(mc.model, "_fem_field", None) == "node" for mc in _model_calls(node))


def _sample_grid_direct(coeff_node, grid, nz, period, z_range, params=None):
    """Evaluate an analytic coefficient expression directly on the RCWA grid (exact, no mesh noise).

    Used when the permittivity is an analytic function of the coordinates (no per-node field); this
    avoids the staircase/interp noise that makes z-slices look non-invariant on an unstructured mesh.
    ``params`` substitutes trainable-parameter values (so the eager structure/k0 inference uses valid
    values, e.g. a K0 parameter)."""
    xs = np.linspace(0, period[0], grid, endpoint=False)
    ys = np.linspace(0, period[1], grid, endpoint=False)
    zs = np.linspace(z_range[0], z_range[1], nz)
    GX, GY, GZ = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], 1)
    vals = np.asarray(_eval_coeff_points(coeff_node, pts, params or {})).reshape(grid, grid, nz)
    return np.moveaxis(vals, 2, 0).astype(complex), zs  # -> (nz, grid, grid)


def _sample_grid(domain, eps_nodes, grid, nz, period, z_range):
    """Interpolate nodal eps onto an (nz, grid, grid) unit-cell array over the z-extent."""
    from scipy.interpolate import griddata

    pts = np.asarray(domain.points)
    xs = np.linspace(0, period[0], grid, endpoint=False)
    ys = np.linspace(0, period[1], grid, endpoint=False)
    zs = np.linspace(z_range[0], z_range[1], nz)
    GX, GY = np.meshgrid(xs, ys, indexing="ij")
    E = np.empty((nz, grid, grid), complex)
    for k, zz in enumerate(zs):
        sel = np.abs(pts[:, 2] - zz) < (z_range[1] - z_range[0]) / (nz - 1) * 0.75 + 1e-9
        if sel.sum() < 3:
            j = np.argsort(np.abs(pts[:, 2] - zz))[: max(3, grid)]
            sel = np.zeros(len(pts), bool)
            sel[j] = True
        vals = griddata(pts[sel][:, :2], eps_nodes[sel], (GX, GY), method="nearest")
        E[k] = vals
    return E, zs


def _source_kin(femobj, domain, params=None):
    """Read the illuminated face and transverse wavevector k_in from the assembled forcing b.
    ``params`` overrides trainable-parameter values used when assembling (e.g. a K0/source parameter)."""
    op = femobj.operator
    if not (isinstance(op, tuple) and len(op) == 2):
        raise RcwaError(
            "jno.rcwa expects a complex (time-harmonic) Helmholtz problem: fem.operator did not return a "
            "(real, imag) operator pair. Author the field with complex=True."
        )
    opr, opi = op
    if hasattr(opr, "evaluate"):  # lazy (parametric) operator
        pz = _param_values(femobj, femobj._constraints[0] if femobj._constraints else 0)
        pz.update({k: jnp.asarray(v) for k, v in (params or {}).items()})
        _, br = opr.evaluate(pz)
        _, bi = opi.evaluate(pz)
    else:  # eager (parameter-free) (A, b) legs
        br, bi = opr[1], opi[1]
    b = np.asarray(br).reshape(-1) + 1j * np.asarray(bi).reshape(-1)
    nz = np.where(np.abs(b) > 1e-9 * (np.abs(b).max() + 1e-30))[0]
    if len(nz) == 0:
        raise RcwaError(
            "no source/forcing found in the problem: nothing to illuminate the cell with. Author an "
            "incident wave (e.g. an inhomogeneous absorbing boundary carrying the incoming field)."
        )
    pts = np.asarray(domain.points)
    zc = pts[nz, 2]
    face_z = np.median(zc)
    if np.ptp(zc) > 1e-6 * (pts[:, 2].max() - pts[:, 2].min() + 1e-30) + 1e-9:
        raise RcwaError(
            f"the forcing is spread over a z-range [{zc.min():.3g},{zc.max():.3g}], not a single "
            f"illuminated face; RCWA needs the incident wave on one ambient face."
        )
    # transverse angle from the phase gradient of b across the lit face
    xy = pts[nz, :2]
    phase = np.unwrap(np.angle(b[nz]))
    A = np.c_[xy, np.ones(len(xy))]
    (kx, ky, _), *_ = np.linalg.lstsq(A, phase, rcond=None)
    k_in = (float(kx), float(ky))
    if abs(kx) < 1e-6 and abs(ky) < 1e-6:
        k_in = (0.0, 0.0)
    return face_z, k_in


def _cell_grid_at_z(period, grid, z):
    xs = np.linspace(0, period[0], grid, endpoint=False)
    ys = np.linspace(0, period[1], grid, endpoint=False)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), np.full(grid * grid, z)], 1)


def _eval_coeff_points(coeff_node, pts, params):
    """Evaluate the permittivity coefficient at ``pts``, substituting trainable parameters from
    ``params`` (jax values). Returns a JAX array -- differentiable in the design parameters."""
    import equinox as eqx

    from jno.trace import Variable
    from jno.trace_evaluator import TraceEvaluator

    table = {}
    for mc in _model_calls(coeff_node):
        mod, name = mc.model.module, getattr(mc.model, "_parameter_name", None)
        if name is not None and name in params:
            mod = eqx.tree_at(lambda m: m.value, mod, jnp.asarray(params[name]))
        table[mc.model.layer_id] = mod
    tags = {v.tag for v in _walk_nodes(coeff_node) if isinstance(v, Variable)}
    ctx = {t: jnp.asarray(pts) for t in tags} if tags else {}
    return TraceEvaluator(table).evaluate(coeff_node, context=ctx)


class _RcwaProblem:
    """An RCWA problem inferred from a jNO constraint list / FEM problem. Holds the inferred
    :class:`RcwaSpec` and builds the fmmax engine on :meth:`solve`."""

    def __init__(self, spec, orders, formulation="JONES_DIRECT_FOURIER", resample=None):
        self.spec = spec
        self.orders = orders
        self.formulation = formulation
        self._resample = resample  # params -> [(thickness, eps_grid), ...]  (jax; for differentiable solves)

    def __repr__(self):
        return f"_RcwaProblem(orders={self.orders}, {self.spec!r})"

    def _engine(self):
        return Rcwa(
            self.spec.layers,
            period=self.spec.period,
            orders=self.orders,
            wavelength=self.spec.wavelength,
            k_in=self.spec.k_in,
            formulation=self.formulation,
            assume_periodic=True,
        )

    def solve(self, params=None, wavelength=None):
        """Build the fmmax engine and solve.

        Pass ``params={name: value}`` (JAX values for the trainable ``jno.np.parameter`` coefficients) to
        get a **differentiable** solve: the permittivity is re-sampled from those values and the whole
        modal solve traces, so ``jax.grad`` of a ``sol.efficiency(...)`` objective flows to the design.

        Pass ``wavelength`` to override the inferred one -- sweep it for a **broadband / dispersion**
        response (the transverse wavevector ``k_in`` is held fixed, i.e. fixed-momentum dispersion). The
        result is differentiable in the wavelength too."""
        eng = self._engine()
        if params is None:
            return eng.solve(wavelength=wavelength)
        if self._resample is None:
            raise RcwaError(
                "differentiable solve(params=...) needs the analytic re-sampling path, which is only built "
                "for an analytic permittivity (no per-node field). This problem uses a nodal-field "
                "permittivity; differentiable nodal re-sampling is not implemented yet."
            )
        layers, wl = self._resample(params)  # eps AND wavelength re-derived from the parameters
        return eng.solve(layers=layers, wavelength=wavelength if wavelength is not None else wl)


def rcwa(problem, *, orders, wavelength=None, grid=64, nz=64, slices=None, params=None, formulation="JONES_DIRECT_FOURIER"):
    """Infer and build an RCWA problem from a jNO constraint list (or built ``FEM``).

    Everything is read out of the traced problem: **periodicity + period** (Floquet ties — absent ⇒
    raise), the **super/substrate ambients**, the **permittivity** (the ``K0**2*eps`` coefficient
    recovered from the scalar Helmholtz volume term, sampled along z), the **wavelength** (``k0`` from
    the vacuum superstrate, unless ``wavelength`` is given), and the **incident wave** (illuminated face
    + transverse angle ``k_in``, from the assembled forcing).

    Parameters
    ----------
    problem:
        The same constraint list you would pass to :func:`jno.fem`, or an already-built ``FEM``.
    orders:
        Fourier truncation for the modal solve.
    wavelength:
        Free-space wavelength (same length unit as the geometry). Optional — inferred from the
        superstrate (assumed vacuum) when omitted. Pass it to override or if no ambient is vacuum.
    grid, nz:
        Transverse resolution and number of z-samples used to detect the layer stack.
    slices:
        Staircase a continuously-varying permittivity into this many layers instead of raising.

    Returns
    -------
    _RcwaProblem
        Holds the inferred :class:`RcwaSpec` (``.spec``) and a :meth:`~_RcwaProblem.solve`.
    """
    femobj = _as_fem(problem)
    domain = femobj.domain
    axes, period = _periodic_period(femobj, domain)
    bottom, top = _z_ambient_faces(domain)
    pts = np.asarray(domain.points)
    z_range = (float(pts[:, 2].min()), float(pts[:, 2].max()))

    # permittivity coefficient K0^2 * eps, recovered from the volume weak form (no assembly, no eps arg)
    coeff_node = _extract_permittivity_coeff(_volume_constraint(femobj))
    # construction-time parameter values: the trainable parameters' current values, overridden by any
    # `params=` the caller supplied -- so the eager inference (structure, source, k0) uses valid values
    # (e.g. a K0 parameter that must be non-zero for the source to exist).
    cparams = _param_values(femobj, coeff_node)
    cparams.update({k: jnp.asarray(v) for k, v in (params or {}).items()})
    if _has_nodal_field(coeff_node):  # per-node design field -> interpolate off the mesh
        coeff_nodes = _eval_expr_nodes(coeff_node, domain)
        if np.iscomplexobj(coeff_nodes) and np.max(np.abs(coeff_nodes.imag)) < 1e-9:
            coeff_nodes = coeff_nodes.real.astype(complex)
        C, zs = _sample_grid(domain, coeff_nodes, grid, nz, period, z_range)
        zmids = None  # nodal-field path: no analytic re-sampling closure (differentiable solve unsupported)
    else:  # analytic permittivity -> sample the grid exactly
        C, zs = _sample_grid_direct(coeff_node, grid, nz, period, z_range, cparams)
    coeff_layers = detect_layers(C, zs, slices=slices)
    zmids = detect_layers.last_zmid if not _has_nodal_field(coeff_node) else None

    face_z, k_in = _source_kin(femobj, domain, cparams)
    source_at_bottom = abs(face_z - _region_centroid(domain, bottom)[2]) < abs(face_z - _region_centroid(domain, top)[2])
    source_face = bottom if source_at_bottom else top

    # orient so the incident (source-side) ambient is layers[0] = superstrate
    if not source_at_bottom:
        coeff_layers = list(reversed(coeff_layers))
        if zmids is not None:
            zmids = list(reversed(zmids))
    # k0 from the superstrate (vacuum ⇒ coeff = k0^2), unless wavelength is given
    super_coeff = float(np.real(coeff_layers[0][1]).mean())
    if wavelength is not None:
        k0 = 2 * np.pi / float(wavelength)
    elif super_coeff <= 0:
        raise RcwaError(
            "cannot infer wavelength: the superstrate permittivity coefficient is non-positive; pass wavelength=."
        )
    else:
        k0 = float(np.sqrt(super_coeff))
    k0sq = k0 * k0
    layers = [(t, np.asarray(e) / k0sq) for t, e in coeff_layers]  # relative permittivity

    # differentiable re-sampling: re-evaluate the analytic permittivity at each layer's representative z
    # from a parameter dict, so a design (jno.np.parameter) flows through solve(params=...). Only for the
    # analytic path (nodal-field re-sampling would need mesh interpolation -> deferred, raises in solve).
    resample = None
    if zmids is not None:
        thicks = [t for t, _ in coeff_layers]

        def resample(params):
            # Re-derive EVERYTHING RCWA reads from the (parameterized) permittivity coefficient K0^2*eps:
            #   * k0 = sqrt(coeff in the vacuum superstrate)  -> a WAVELENGTH parameter flows here
            #   * each layer's relative eps = coeff(z)/k0^2    -> an EPS/shape parameter flows here
            # so a jno.np.parameter anywhere in the volume weak form is a differentiable knob.
            sup = jnp.mean(jnp.real(_eval_coeff_points(coeff_node, _cell_grid_at_z(period, grid, zmids[0]), params)))
            k0 = jnp.sqrt(sup)
            out = []
            for thick, zmid in zip(thicks, zmids):
                cvals = jnp.reshape(
                    _eval_coeff_points(coeff_node, _cell_grid_at_z(period, grid, zmid), params), (grid, grid)
                )
                out.append((thick, jnp.real(cvals) / (k0 * k0)))
            return out, 2 * jnp.pi / k0

    spec = RcwaSpec(
        period=period,
        layers=layers,
        wavelength=2 * np.pi / k0,
        k_in=k_in,
        source_face=source_face,
        ambient_faces=(bottom, top),
        periodic_axes=axes,
    )
    return _RcwaProblem(spec, orders=orders, formulation=formulation, resample=resample)
