"""Electro-thermal coupling: a ``jno.peec`` network and the ``jno.fem`` problem of its temperature.

Written as two term lists and solved together by ``jno.core``::

    T, s = d.fem_symbols()
    d.attach(sigma=sigma20 / (1 + alpha * (T - 293.15)), k=400.0)   # sigma depends on the FEM field
    em = jno.peec([v(P) - v(N) - 1.0], freq=1e6)
    heat = jno.fem([d.k * inner(grad(T), grad(s)) - em.loss * s, T(N) - 300.0])
    sol = jno.core([em, heat]).solve()

The coupling is a fixed point on the FEM field ``T``: the network is solved with the conductivity at the
current temperature, its loss density per conductor (``em.loss``) is the heat source of the FEM problem,
and the FEM solve gives the next temperature. The temperature reaches each PEEC element by linear (P1)
interpolation of the FEM field at the element centre.

Differentiability. Every step of the map is JAX -- the conductivity is evaluated by the trace evaluator,
the network is the frozen ``BuiltPEEC``, the heat solve is the assembled ``FemLinearSystem`` -- so when a
trainable ``jno.np.parameter`` sits in either problem, ``solve()`` returns a trace node over those
parameters and the gradient flows through the converged fixed point by ``jax.lax.custom_root`` (implicit
differentiation, never unrolled iterations). Without a trainable parameter it returns the converged
:class:`ElectroThermalSolution` directly.

Scope, up front:

* the heat problem is **linear** in its field (a constant or region-wise ``k``); a nonlinear one is refused;
* the field is a **scalar P1** field on the same domain as the network;
* the network is solved at **one** frequency;
* the geometry is frozen at the first solve, as for :meth:`jno.peec.PEEC.build`.
"""

from __future__ import annotations

import numpy as np


def _is_peec(obj):
    from ...peec import PEEC

    return isinstance(obj, PEEC)


def _is_fem(obj):
    return hasattr(obj, "_op") and hasattr(obj, "operator") and not _is_peec(obj)


def detect_pair(constraints):
    """``(em, heat)`` when ``constraints`` is one ``jno.peec`` network and one ``jno.fem`` problem, else ``None``."""
    try:
        items = list(constraints)
    except TypeError:
        return None
    if len(items) != 2:
        return None
    ems = [c for c in items if _is_peec(c)]
    fems = [c for c in items if _is_fem(c)]
    if len(ems) == 1 and len(fems) == 1:
        return ems[0], fems[0]
    if ems:
        raise ValueError(
            "jno.core: a jno.peec network is coupled to exactly ONE jno.fem problem -- the thermal problem "
            "whose source is `em.loss`. Got "
            f"{[type(c).__name__ for c in items]}."
        )
    return None


class ElectroThermalSolution:
    """The converged coupled state.

    Attributes:
        field: the FEM nodal values of the coupled field (the temperature), ``(n_nodes,)``.
        em: the :class:`~jno.peec.PEECSolution` at the converged conductivity.
        iterations: fixed-point passes taken.
        change: the last relative change of the field, ``max|T_k+1 - T_k| / max|T_k+1|``.
    """

    def __init__(self, field, em, iterations, change):
        self.field, self.em, self.iterations, self.change = field, em, iterations, change

    def __repr__(self):
        return (
            f"ElectroThermalSolution(iterations={self.iterations}, change={self.change:.2e}, "
            f"field range=[{float(np.min(self.field)):.6g}, {float(np.max(self.field)):.6g}])"
        )


class ElectroThermal:
    """The coupled driver behind ``jno.core([em, heat])``. See the module docstring."""

    def __init__(self, em, heat):
        self.em, self.heat = em, heat
        loss = getattr(em, "_loss_params", None)
        if not loss:
            raise ValueError(
                "jno.core([em, heat]): the FEM problem does not use the network's loss. Write it as the heat "
                "source of the FEM problem -- `jno.fem([... - em.loss * s, ...])` -- so the two are coupled."
            )
        from ...trace import FemLinearSystem

        op = heat._op
        if not isinstance(op, FemLinearSystem) or not op.is_parametric:
            raise NotImplementedError(
                "jno.core([em, heat]): the heat problem must be a LINEAR jno.fem problem whose source is "
                "`em.loss`. A nonlinear heat problem (a temperature-dependent k, radiation) is not supported "
                "by the coupled solve yet."
            )
        names = list(op.runtime_parameter_exprs)
        self._loss_names = {f"__peec_loss_{r}": r for r in loss}
        missing = [n for n in self._loss_names if n not in names]
        if missing:
            raise ValueError(
                "jno.core([em, heat]): the FEM problem does not use the network's loss. Write it as the heat "
                "source of the FEM problem -- `jno.fem([... - em.loss * s, ...])` -- so the two are coupled."
            )
        if np.size(em.freq) != 1:
            raise NotImplementedError(
                "jno.core([em, heat]): the network is solved at several frequencies, and a heat source is one "
                "loss. Give jno.peec a single frequency for the coupled solve."
            )
        self._op = op
        self._fem_param_names = [n for n in names if n not in self._loss_names]
        self._fem_param_nodes = [op.runtime_parameter_exprs[n] for n in self._fem_param_names]
        self._built = None

    # -- set-up: the FEM side, the reference field, the frozen network, the interpolation -------------------
    def _fem_solve(self, loss_values, fem_values):
        import jax.numpy as jnp

        from .linear import sparse_lu_solve

        vals = {n: jnp.reshape(jnp.real(jnp.asarray(loss_values.get(r, 0.0))), (1,)) for n, r in self._loss_names.items()}
        vals.update(dict(zip(self._fem_param_names, fem_values)))
        A, b = self._op.evaluate(vals)
        return sparse_lu_solve(A, jnp.asarray(b).reshape(-1))

    def _setup(self, fem_values):
        import jax.numpy as jnp

        from ...peec import _field_symbols
        from .fem_adapt import _locate_barycentric

        d = self.heat.domain if hasattr(self.heat, "domain") else self.em.domain
        T0 = self._fem_solve({}, fem_values)  # the field with no loss: the reference for the structural build
        pts = np.asarray(self.heat.points)
        cells = np.asarray(d._cells_p1())
        if pts.shape[0] != int(np.max(cells)) + 1 or np.asarray(T0).reshape(-1).shape[0] != pts.shape[0]:
            raise NotImplementedError(
                "jno.core([em, heat]): the coupled field must be a scalar P1 field on the network's domain "
                f"(one value per mesh node); this one has {np.asarray(T0).size} values on {pts.shape[0]} nodes."
            )
        self.em._field_reference = float(np.mean(np.asarray(T0)))
        B = self.em.build()
        field_exprs = dict(getattr(self.em, "_field_exprs", {}) or {})
        fields = {f for e in field_exprs.values() for f in _field_symbols(e)}
        if len(fields) > 1:
            raise NotImplementedError(
                "jno.core([em, heat]): a conductivity depends on more than one FEM field "
                f"({sorted(f.name for f in fields)}); the coupled solve couples one field."
            )
        # Each field-dependent conductor's element centres, read off the resolver itself (so the order is
        # exactly the order it consumes values in), then the P1 stencil of the FEM mesh at those centres.
        seen = {}

        def _recorder(name):
            def rec(x, y, z):
                seen[name] = np.stack([np.asarray(x), np.asarray(y), np.asarray(z)], axis=1)
                return jnp.ones(np.shape(x))

            return rec

        if field_exprs:
            B._resolve({n: _recorder(n) for n in field_exprs})
        dim = pts.shape[1]
        stencils = {}
        for n, xyz in seen.items():
            idx, w, _inside = _locate_barycentric(pts, cells, xyz[:, :dim], tol=1e-9, k=12)
            stencils[n] = (jnp.asarray(idx), jnp.asarray(w))
        self._built, self._field_exprs, self._field = B, field_exprs, next(iter(fields), None)
        self._stencils = stencils
        self._T0 = jnp.asarray(T0).reshape(-1)

    # -- one pass of the coupling ----------------------------------------------------------------------------
    def _sigma(self, T, em_values):
        from ...peec import _eval_field_material, _eval_material

        B = self._built
        params = dict(zip(self._em_param_names, em_values))
        sig = {}
        for n, expr in self._field_exprs.items():
            idx, w = self._stencils[n]
            T_elem = (w * T[idx]).sum(axis=1)
            sig[n] = _eval_field_material(expr, {self._field: T_elem}, params)
        for n, expr in B._param_exprs.items():
            if n not in sig:
                sig[n] = _eval_material(expr, params)
        return sig

    def _step(self, T, em_values, fem_values):
        import jax.numpy as jnp

        sol = self._built.solve(sigma=self._sigma(T, em_values))
        q = {r: jnp.real(jnp.asarray(v)).reshape(()) for r, v in sol.dissipation().items()}
        return self._fem_solve(q, fem_values), sol

    # -- the solve -------------------------------------------------------------------------------------------
    def solve(self, *, tol: float = 1e-8, max_iter: int = 50):
        """Converge the coupled field. Returns an :class:`ElectroThermalSolution`, or -- when a trainable
        ``jno.np.parameter`` sits in either problem -- a trace node of the converged FEM field over them."""
        from ...peec import _initial_value, _parameters_in

        em_params = _parameters_in(list(self._em_material_exprs().values()))
        self._em_param_names = list(em_params)
        em_nodes = [em_params[n] for n in self._em_param_names]
        if em_nodes or self._fem_param_nodes:
            if self._built is None:
                # The structural build reads concrete values, so it runs here, eagerly, at the parameters'
                # initial values -- never inside the traced solve.
                self._setup([_initial_value(mc) for mc in self._fem_param_nodes])
            return self._parametric(em_nodes, tol=tol, max_iter=max_iter)
        em_vals = [_initial_value(mc) for mc in em_nodes]
        if self._built is None:
            self._setup([])
        T = self._T0
        change, it = float("inf"), 0
        for it in range(1, max_iter + 1):
            Tn, sol = self._step(T, em_vals, [])
            change = float(
                np.max(np.abs(np.asarray(Tn) - np.asarray(T))) / max(float(np.max(np.abs(np.asarray(Tn)))), 1e-300)
            )
            T = Tn
            if change < tol:
                break
        else:
            raise RuntimeError(
                f"jno.core([em, heat]): the electro-thermal fixed point did not converge in {max_iter} passes "
                f"(last relative change {change:.2e} against tol={tol:g}). A loss that grows faster with "
                "temperature than the heat path can remove has no steady state -- thermal runaway -- so check "
                "the drive and the heat sink before raising max_iter."
            )
        _T, sol = self._step(T, em_vals, [])
        return ElectroThermalSolution(np.asarray(T), sol, it, change)

    def _em_material_exprs(self):
        """Every attached conductivity of the network, including a default, for parameter discovery."""
        try:
            sig = dict(self.em.domain.attached("sigma"))
        except KeyError:
            sig = {}
        return sig

    def _parametric(self, em_nodes, *, tol, max_iter):
        import jax
        import jax.numpy as jnp

        from ...trace import FunctionCall
        from .newton_krylov import bicgstab

        n_em = len(em_nodes)
        nodes = list(em_nodes) + list(self._fem_param_nodes)

        def _run(*values):
            em_vals, fem_vals = list(values[:n_em]), list(values[n_em:])

            def phi(T):
                return self._step(T, em_vals, fem_vals)[0]

            def f(T):
                return T - phi(T)

            def forward(_f, T0):
                def cond(s):
                    _T, r, k = s
                    return (r > tol) & (k < max_iter)

                def body(s):
                    T, _r, k = s
                    Tn = phi(T)
                    return Tn, jnp.max(jnp.abs(Tn - T)) / jnp.maximum(jnp.max(jnp.abs(Tn)), 1e-300), k + 1

                T, _r, _k = jax.lax.while_loop(cond, body, (T0, jnp.asarray(jnp.inf), 0))
                return T

            bicg = lambda mv, rr: bicgstab(mv, rr, tol=1e-11, maxit=10000)  # noqa: E731
            tangent = lambda g, y: jax.lax.custom_linear_solve(g, y, bicg, transpose_solve=bicg)  # noqa: E731
            return jax.lax.custom_root(f, jax.lax.stop_gradient(self._T0), forward, tangent)

        node = FunctionCall(_run, nodes, name="electrothermal_solve")
        node._domain = self.em.domain
        return node
