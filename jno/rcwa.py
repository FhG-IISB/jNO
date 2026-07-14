"""Rigorous Coupled-Wave Analysis (RCWA / Fourier Modal Method) for periodic, layered structures.

RCWA is semi-analytic in the propagation direction: each layer is expanded in a truncated Fourier
basis in-plane and solved by an eigenmode decomposition, then layers are stitched with a scattering
matrix. For **periodic, piecewise-z-invariant** structures — extruded metasurface pillars are the
canonical case — this is far cheaper than a full 3-D volumetric solve.

Two entry points, both on `jno.rcwa`:

* **From a jNO problem (the front door).** Hand it the same constraint list you would give
  :func:`jno.fem` (or the already-built ``FEM``) together with the permittivity expression ``eps``;
  RCWA reads the rest out of the traced problem::

      rc  = jno.rcwa(constraints, eps, orders=300, wavelength=1.0)
      sol = rc.solve()                     # period, layers, ambients, source, incidence all inferred
      rc.spec                              # the inferred RcwaSpec (available without fmmax)

  Inferred from the problem: **periodicity + period** (from the Floquet ties — absent ⇒ raise, never
  assumed), the **super/substrate ambients** (the z-normal radiation faces), the **layer stack**
  (``eps`` sampled along z, then :func:`detect_layers`), and the **incident wave** (illuminated face +
  transverse angle ``k_in``, read from the assembled forcing). ``eps`` is passed explicitly because
  isolating one coefficient sub-tree from an assembled weak form is not supported yet; ``wavelength``
  is passed because a bare-float ``k0`` is not an identifiable node in the trace.

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

import numpy as np

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

    layers, report = [], []
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
        report.append(
            f"  layer {i}: z=[{z[a]:.3f},{z[min(b - 1, Nz - 1)]:.3f}] {kind} eps~[{eps_xy.min():.2f},{eps_xy.max():.2f}]"
        )
    detect_layers.last_report = "detected layers:\n" + "\n".join(report)
    return layers


# =====================================================================================
# The forward engine (explicit layers) — the fmmax backend the front door constructs.
# =====================================================================================
class _Sol:
    def __init__(self, fm, s, layers, expansion, nt, Pin, wavelength):
        self._fm, self._s, self._layers, self._ex = fm, s, layers, expansion
        self._nt, self._Pin, self._wl = nt, Pin, wavelength
        self._fwd = np.zeros((2 * nt, 1), complex)

    def _flux(self, amps, layer, backward=False):
        f, b = self._fm.directional_poynting_flux(
            amps if not backward else np.zeros_like(amps),
            amps if backward else np.zeros_like(amps),
            layer,
        )
        return np.asarray(b if backward else f).reshape(-1)  # SIGNED — energy-correct

    def efficiency(self, kind):
        """Total transmitted (``"T"``) or reflected (``"R"``) power fraction over propagating orders."""
        if kind == "T":
            return float(np.sum(self._flux(self._s.s11 @ self._fwd, self._layers[-1]))) / self._Pin
        if kind == "R":
            return float(-np.sum(self._flux(self._s.s21 @ self._fwd, self._layers[0], backward=True))) / self._Pin
        raise RcwaError(f"efficiency(kind): kind must be 'T' or 'R', got {kind!r}")

    def order(self, m, n):
        """Diffraction efficiency into transmitted order ``(m, n)`` (raises if outside the truncation)."""
        bc = np.asarray(self._ex.basis_coefficients)
        hit = np.where((bc[:, 0] == m) & (bc[:, 1] == n))[0]
        if len(hit) == 0:
            raise RcwaError(f"order ({m},{n}) is outside the truncation ({self._nt} terms); raise orders=.")
        i = int(hit[0])
        tf = np.abs(
            np.asarray(
                self._fm.directional_poynting_flux(
                    self._s.s11 @ self._fwd, np.zeros((2 * self._nt, 1), complex), self._layers[-1]
                )[0]
            ).reshape(-1)
        )
        return float(tf[i] + tf[i + self._nt]) / self._Pin


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

    def solve(self, inc=None, wavelength=None, k_in=None):
        """Solve the stack and return a :class:`_Sol`. Raises if the wavelength is unknown or energy
        is not conserved."""
        fm = self.fm
        wl = wavelength if wavelength is not None else self.wavelength
        if wl is None:
            raise RcwaError(
                "wavelength is unknown: pass wavelength= to solve() or at construction; it sets every "
                "layer's eigenmodes and is never defaulted."
            )
        kin = np.asarray(self.k_in if k_in is None else k_in, float)
        lv = fm.LatticeVectors(u=np.array([self.period[0], 0.0]), v=np.array([0.0, self.period[1]]))
        ex = fm.generate_expansion(lv, approximate_num_terms=self.orders)
        nt = ex.num_terms

        def solve_layer(e):
            eg = np.asarray(e) + 0j
            if eg.ndim == 0:
                eg = np.full((1, 1), eg)
            return fm.eigensolve_isotropic_media(np.asarray(wl), kin, lv, eg, ex, formulation=self.formulation)

        layers = [solve_layer(e) for _, e in self.layers_spec]
        thick = [np.asarray(1.0 if t is None or t == np.inf else t) for t, _ in self.layers_spec]
        s = fm.stack_s_matrix(layers, thick)
        fwd = np.zeros((2 * nt, 1), complex)
        fwd[0, 0] = 1.0
        Pin = float(np.sum(np.asarray(fm.directional_poynting_flux(fwd, np.zeros_like(fwd), layers[0])[0])))
        sol = _Sol(fm, s, layers, ex, nt, Pin, wl)
        sol._fwd = fwd
        T, R = sol.efficiency("T"), sol.efficiency("R")
        if T + R > 1.0 + 5e-3:
            raise RcwaError(
                f"energy not conserved: T+R={T + R:.4f} > 1 at orders={self.orders}. The modal solve is "
                f"not converged (raise orders) or the structure is unphysical."
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


def _param_zeros(femobj, eps):
    """{parameter_name: zeros(shape)} for every trainable parameter the operator references."""
    out = {}
    nodes = list(_model_calls(eps))
    for c in getattr(femobj, "_constraints", []):
        nodes += _model_calls(c)
    for mc in nodes:
        m = getattr(mc, "model", None)
        name = getattr(m, "_parameter_name", None)
        if name is None:
            continue
        import jax.numpy as jnp

        out[name] = jnp.zeros(m.module.value.shape, m.module.value.dtype)
    return out


def _eval_eps_nodes(eps, domain, params):
    """Evaluate the eps expression at every mesh node (where nodal parameters align)."""
    import jax.numpy as jnp

    from jno.trace import Variable
    from jno.trace_evaluator import TraceEvaluator

    node_coords = jnp.asarray(np.asarray(domain.points))
    table = {}
    for mc in _model_calls(eps):
        m = mc.model
        mod = m.module
        name = getattr(m, "_parameter_name", None)
        if name is not None and name in params:
            import equinox as eqx

            mod = eqx.tree_at(lambda mm: mm.value, mod, jnp.asarray(params[name]))
        table[m.layer_id] = mod
    tags = {v.tag for v in _walk_nodes(eps) if isinstance(v, Variable)}
    ctx = {t: node_coords for t in tags} if tags else {}
    return np.asarray(TraceEvaluator(table).evaluate(eps, context=ctx)).reshape(-1)


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


def _source_kin(femobj, domain):
    """Read the illuminated face and transverse wavevector k_in from the assembled forcing b."""
    op = femobj.operator
    if not (isinstance(op, tuple) and len(op) == 2):
        raise RcwaError(
            "jno.rcwa expects a complex (time-harmonic) Helmholtz problem: fem.operator did not return a "
            "(real, imag) operator pair. Author the field with complex=True."
        )
    opr, opi = op
    pz = _param_zeros(femobj, femobj._constraints[0] if femobj._constraints else 0)
    _, br = opr.evaluate(pz)
    _, bi = opi.evaluate(pz)
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


class _RcwaProblem:
    """An RCWA problem inferred from a jNO constraint list / FEM problem. Holds the inferred
    :class:`RcwaSpec` and builds the fmmax engine on :meth:`solve`."""

    def __init__(self, spec, orders, formulation="JONES_DIRECT_FOURIER"):
        self.spec = spec
        self.orders = orders
        self.formulation = formulation

    def __repr__(self):
        return f"_RcwaProblem(orders={self.orders}, {self.spec!r})"

    def solve(self):
        """Build the fmmax engine from the inferred spec and solve (needs the fmmax backend)."""
        eng = Rcwa(
            self.spec.layers,
            period=self.spec.period,
            orders=self.orders,
            wavelength=self.spec.wavelength,
            k_in=self.spec.k_in,
            formulation=self.formulation,
            assume_periodic=True,
        )
        return eng.solve()


def rcwa(problem, eps, *, orders, wavelength, grid=64, nz=64, slices=None, formulation="JONES_DIRECT_FOURIER"):
    """Infer and build an RCWA problem from a jNO constraint list (or built ``FEM``) plus ``eps``.

    Parameters
    ----------
    problem:
        The same constraint list you would pass to :func:`jno.fem`, or an already-built ``FEM``.
    eps:
        The permittivity expression (the traced coefficient, may depend on a trainable parameter).
    orders:
        Fourier truncation for the modal solve.
    wavelength:
        Free-space wavelength (same length unit as the geometry). Passed explicitly — a bare-float
        ``k0`` is not an identifiable node in the trace.
    grid, nz:
        Transverse resolution and number of z-samples used to detect the layer stack.
    slices:
        Staircase a continuously-varying ``eps`` into this many layers instead of raising.

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

    params = _param_zeros(femobj, eps)
    eps_nodes = _eval_eps_nodes(eps, domain, params)
    if np.iscomplexobj(eps_nodes) and np.max(np.abs(eps_nodes.imag)) < 1e-9:
        eps_nodes = eps_nodes.real.astype(complex)
    E, zs = _sample_grid(domain, eps_nodes, grid, nz, period, z_range)
    layers = detect_layers(E, zs, slices=slices)

    face_z, k_in = _source_kin(femobj, domain)
    source_face = (
        bottom
        if abs(face_z - _region_centroid(domain, bottom)[2]) < abs(face_z - _region_centroid(domain, top)[2])
        else top
    )

    spec = RcwaSpec(
        period=period,
        layers=layers,
        wavelength=float(wavelength),
        k_in=k_in,
        source_face=source_face,
        ambient_faces=(bottom, top),
        periodic_axes=axes,
    )
    return _RcwaProblem(spec, orders=orders, formulation=formulation)
