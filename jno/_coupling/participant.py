"""The problems of a partitioned coupling, one adapter per solver kind (``jno.fem``, ``jno.fdm``, ``jno.fvm``).

Based on:
    Bungartz, H.-J. et al. (2016). preCICE - A fully parallel library for multi-physics surface coupling.
    Computers & Fluids 141, 250-258 (the participant / interface-data model).

A participant answers the questions the coupling driver asks of every problem, whatever its discretisation:

- which unknowns are its own (``owns``), so an interface condition can be split by who it belongs to;
- the interface a boundary tag of its mesh carries (``interface``);
- a *data field* on its mesh, a nodal field the driver fills between solves with what the other participants
  said (``data``), and the problem rebuilt with conditions that read it (``rebuild``);
- a solve with given data (``solve``);
- the value of one of its own expressions at its interface nodes (``sample``): a value is evaluated there, a
  flux is read from the discretisation's own discrete flux (the boundary-face rule of a finite volume, the
  boundary stencil of a finite difference, the consistent reaction of a finite element).
"""

from __future__ import annotations

import numpy as np

from .._strong_form import _iter, _map_children, _unwrap
from .transfer import Interface


def _derivative_nodes():
    from ..trace import Hessian, Jacobian

    return (Jacobian, Hessian)


def has_derivative(expr, owns) -> bool:
    """Does ``expr`` differentiate an unknown that ``owns`` claims (a flux-type expression)?"""

    def walk(n):
        n = _unwrap(n)
        if isinstance(n, _derivative_nodes()) and mentions(n.target, owns):
            return True
        return any(walk(c) for c in _iter(n))

    return walk(expr)


def mentions(expr, owns) -> bool:
    """Does ``expr`` contain an unknown that ``owns`` claims?"""

    def walk(n):
        n = _unwrap(n)
        if owns(n):
            return True
        return any(walk(c) for c in _iter(n))

    return walk(expr)


def evaluate_at(expr, points, leaf, normals=None, t=None, domain=None):
    """``expr`` at ``points`` ``(m, dim)``: every unknown ``leaf`` claims is replaced by the ``(m, c)`` values it
    returns, the spatial coordinates read ``points``, a normal reads ``normals`` and the time ``t``. For value
    expressions (no derivative of an unknown) of any participant, at its interface nodes."""
    import jax.numpy as jnp

    from .._fvm.trace import _is_normal
    from .._strong_form import _context_leaf, _iter_variables
    from ..trace_evaluator import TraceEvaluator

    m = int(points.shape[0])
    leaves = {}

    def rewrite(node):
        n = _unwrap(node)
        value = leaf(n)
        if value is not None:
            tag = f"cpl_leaf_{len(leaves)}"
            leaves[tag] = jnp.asarray(value).reshape(m, -1)
            return _context_leaf(tag, None, domain)
        if isinstance(n, _derivative_nodes()):
            raise NotImplementedError("coupling: a derivative here is read from the discretisation, not evaluated.")
        return _map_children(n, rewrite)

    expr = rewrite(expr)
    context = {"__time__": jnp.full((m, 1), 0.0 if t is None else t)}
    for v in _iter_variables(expr):
        if v.tag in leaves or v.tag in context:
            continue
        if _is_normal(v):
            if normals is None:
                raise ValueError("coupling: this expression reads a normal, which the interface does not carry.")
            context[v.tag] = jnp.asarray(normals)
        elif getattr(v, "axis", "spatial") == "spatial":
            context[v.tag] = jnp.asarray(points)
        elif getattr(v, "axis", None) == "temporal":
            context[v.tag] = jnp.full((m, 1), 0.0 if t is None else t)
    context.update(leaves)
    return TraceEvaluator(params={}).evaluate(expr, context=context, var_bindings={})


class Participant:
    """What the driver needs of one problem; subclasses implement it per discretisation."""

    kind = "?"

    def __init__(self, problem):
        self.problem = problem
        self.domain = problem.domain
        self.points = np.asarray(self.domain.mesh_connectivity["points"], dtype=float)
        self.dim = int(getattr(self.domain, "dimension", self.points.shape[1]))
        self.points = self.points[:, : self.dim]
        self._interfaces = {}

    def __repr__(self):
        return f"<{self.kind} participant>"

    # -- structure ------------------------------------------------------------------------------------
    def owns(self, node) -> bool:
        raise NotImplementedError

    def interface(self, tag, measure=None) -> Interface:
        key = (tag, measure if isinstance(measure, (str, type(None))) else id(measure))
        if key not in self._interfaces:
            self._interfaces[key] = Interface.of(self.domain, tag, measure)
        return self._interfaces[key]

    def constraints(self):
        raise NotImplementedError

    def rebuild(self, extra):
        """The problem with the ``extra`` conditions appended to its own."""
        raise NotImplementedError

    # -- data -------------------------------------------------------------------------------------------
    def data(self, name):
        """A new nodal data field on this mesh: ``(node, setter)``. ``node`` goes into a condition; the setter
        takes ``(values, problem)`` (one value per mesh node) and returns what :meth:`solve` needs for it."""
        raise NotImplementedError

    def nodal(self, iface, values):
        """Interface-node values ``(m,)`` spread to a full nodal field (zero off the interface)."""
        out = np.zeros(self.points.shape[0])
        out[iface.nodes] = np.asarray(values).reshape(-1)
        return out

    def essential(self, problem=None) -> bool:
        """Does the problem hold an essential (value) condition of its own? A steady problem without one, given
        only fluxes on its interfaces, is singular (a floating subdomain)."""
        raise NotImplementedError

    def transient(self, problem=None) -> bool:
        raise NotImplementedError

    def autonomous(self) -> bool:
        """Do the problem's own conditions and coefficients leave time alone (no explicit ``t`` outside a time
        derivative)? Then, with its interface data held over a window, a window gives the same result at any
        start time, and one compiled window march serves them all."""
        from ..trace import Hessian, Jacobian, TemporalDerivative, Variable

        def walk(n):
            n = _unwrap(n)
            if isinstance(n, (TemporalDerivative, Jacobian, Hessian)):
                return walk(n.target)
            if isinstance(n, Variable):
                return getattr(n, "axis", None) == "temporal"
            return any(walk(c) for c in _iter(n))

        return not any(walk(c) for c in self.constraints())

    def march(self, problem, state, t0, window, data, time=None):
        """The state after the window ``[t0, t0 + window]`` from ``state``, the data held over it."""
        raise NotImplementedError

    def initial(self, problem):
        """The state its initial condition gives."""
        raise NotImplementedError

    def _span(self, t0, window):
        return (0.0, float(window)) if self.autonomous() else (float(t0), float(t0) + float(window))

    # -- solving and reading back -----------------------------------------------------------------------
    def solve(self, problem, data, x0=None):
        raise NotImplementedError

    def sample(self, problem, state, expr, tag, iface, data=None):
        """``expr`` (one of this participant's own expressions) at the interface nodes, ``(m,)``; ``data`` the
        participant's current interface data (its boundary conditions read them)."""
        raise NotImplementedError


class _StrongForm(Participant):
    """``jno.fdm`` and ``jno.fvm``: unknowns are ``domain.unknown()`` fields, data fields are nodal parameters."""

    def owns(self, node) -> bool:
        from ..trace import ModelCall

        n = _unwrap(node)
        return isinstance(n, ModelCall) and any(n.model is w for w in self.problem.unknowns)

    def constraints(self):
        return list(self.problem._constraints)

    def essential(self, problem=None) -> bool:
        p = problem or self.problem
        return bool(getattr(p, "_dirichlet", None)) or bool(getattr(p, "_pins", None))

    def transient(self, problem=None) -> bool:
        return bool(getattr(problem or self.problem, "_transient", False))

    def data(self, name):
        import jax.numpy as jnp

        import jno

        node = jno.np.parameter((self.points.shape[0],), name=name)  # no optimizer: data, the solve stays eager
        self.__dict__.setdefault("_data_nodes", {})[node.model.layer_id] = node

        def setter(values, problem=None):
            return {node.model.layer_id: jnp.asarray(values)}

        return node, setter

    def _scope(self, problem, data):
        """``extra_params`` for ``data`` (``{layer_id: values}``): each data field's module holding its values."""
        nodes = self.__dict__.get("_data_nodes", {})
        return {lid: problem._trainable_module(nodes[lid], v) for lid, v in data.items()}

    def solve(self, problem, data, x0=None):
        """One steady solve with the data as traced inputs of a solve compiled once per problem: a new value of the
        interface data is a new argument, not a new program (a module swap would change the problem's fingerprint
        and recompile: measured 1.4 s per coupling iteration on a 246-cell slab, against milliseconds)."""
        import jax
        import jax.numpy as jnp

        keys = tuple(sorted(data))
        cache = problem.__dict__.setdefault("_cpl_solve", {})
        fn = cache.get(keys)
        if fn is None:
            from .. import solve as _solve

            def run(values, x0):
                # a direct solve: exact to round-off, so the coupling can converge below an iterative solver's
                # tolerance, and robust on the traced path (matrix-free BiCGStab diverged on fdm flux rows there)
                scope = self._scope(problem, dict(zip(keys, values)))
                return jnp.asarray(problem._steady_solve(x0=x0, extra_params=scope, linear=_solve.lu()))

            fn = cache[keys] = jax.jit(run)
        x0 = jnp.zeros(problem._Ntot) if x0 is None else jnp.asarray(x0).reshape(-1)  # one signature: no retrace
        return fn([data[k] for k in keys], x0).reshape(-1)


class FDMParticipant(_StrongForm):
    """A ``jno.fdm`` problem. Its data fields hold their values (the setter swaps the module): jno.fdm reads its
    data fields live at every call of its compiled solve, and its traced ``extra_params`` route assembles flux
    rows that read data wrongly (NaN on a Neumann condition with injected data)."""

    kind = "fdm"

    def data(self, name):
        import equinox as eqx
        import jax.numpy as jnp

        import jno

        node = jno.np.parameter((self.points.shape[0],), name=name)  # no optimizer: data, the solve stays eager

        def setter(values, problem=None):
            node.model.module = eqx.tree_at(lambda m: m.value, node.model.module, jnp.asarray(values))
            return {}

        return node, setter

    def solve(self, problem, data, x0=None):
        import jax.numpy as jnp

        from .. import solve as _solve

        x0 = None if x0 is None else jnp.asarray(x0).reshape(-1)
        return jnp.asarray(problem._steady_solve(x0=x0, linear=_solve.lu())).reshape(-1)

    def march(self, problem, state, t0, window, data, time=None):
        import jax.numpy as jnp

        traj = problem._march(time=time, x0=jnp.asarray(state).reshape(-1), span=self._span(t0, window))
        return jnp.asarray(traj)[-1].reshape(-1)

    def initial(self, problem):
        import jax.numpy as jnp

        if getattr(problem, "_time_order", 1) == 2:
            raise NotImplementedError("coupling: a u.tt problem marches [u; v]; its windows are not coupled yet.")
        return jnp.asarray(problem._initial_state()).reshape(-1)

    def rebuild(self, extra):
        import jno

        return jno.fdm(self.constraints() + list(extra))

    def sample(self, problem, state, expr, tag, iface, data=None):
        import jax.numpy as jnp

        u = jnp.asarray(state).reshape(-1)
        idx = np.asarray(iface.nodes)
        if has_derivative(expr, self.owns):  # the boundary stencil the flux conditions use
            nodes, _ = problem._node_normals(tag)
            nodes = np.asarray(nodes, dtype=int)
            skeleton = problem._flux_skeleton(expr, nodes)
            k = problem._flux_owner(expr, nodes)
            out = problem._flux_row_fn(skeleton, None, problem._ncomp[k])(u, None)  # data: the stored values
            out = jnp.asarray(out)
            return out[idx] if out.ndim == 1 else out[idx, 0]
        fields = u.reshape(problem._nf, problem._N).T  # (N, nf)

        def leaf(n):
            for k, w in enumerate(problem.unknowns):
                if self.owns(n) and _unwrap(n).model is w:
                    return fields[idx][:, list(problem._blocks_of(k))]
            return None

        return jnp.asarray(evaluate_at(expr, iface.points, leaf, domain=self.domain)).reshape(idx.size, -1)[:, 0]


class FVMParticipant(_StrongForm):
    kind = "fvm"

    def rebuild(self, extra):
        import jno

        return jno.fvm(self.constraints() + list(extra), centering=self.problem.graph.centering)

    def sample(self, problem, state, expr, tag, iface, data=None):
        """Evaluated on the tag's boundary faces with the problem's own boundary rule (face values and normal
        derivatives as its fluxes see them), then lumped to the nodes: ``q_i = Σ_f a_fi F_f / Σ_f a_fi`` with
        ``a_fi = ∫_f φ_i dμ``, so ``Σ_i w_i q_i = Σ_f ∫_f F dμ`` exactly. Compiled once per expression."""
        import jax
        import jax.numpy as jnp

        data = data or {}
        keys = tuple(sorted(data))
        faces = np.asarray(problem._region_faces(tag))
        A = jnp.asarray(self._attachment(problem, faces, iface))
        cache = problem.__dict__.setdefault("_cpl_sample", {})
        fn = cache.get((id(expr), tag, keys))
        if fn is None:

            def run(state, values):
                F = self._face_values(problem, state, expr, faces, dict(zip(keys, values)))
                return (A @ F) / A.sum(axis=1)

            fn = cache[(id(expr), tag, keys)] = jax.jit(run)
        return fn(jnp.asarray(state), [data[k] for k in keys])

    def march(self, problem, state, t0, window, data, time=None):
        """One window, compiled once per window length for an autonomous problem: the data are traced inputs
        (the march reads them through ``_override``). An explicit scheme with ``cfl=`` gets its substeps here,
        from the state at the window's start, as a static number."""
        import copy

        import jax
        import jax.numpy as jnp

        from ..utils.solver.timeschemes import _ERKScheme

        keys = tuple(sorted(data))
        span = self._span(t0, window)
        state = jnp.asarray(state).reshape(-1)
        scheme, sub = time, None
        if isinstance(time, _ERKScheme) and time.cfl is not None:
            sub = self._cfl_substeps(problem, state, time)
            scheme = copy.copy(time)
            scheme.cfl = None

        def run(values, x0):
            problem._override = self._scope(problem, dict(zip(keys, values)))
            try:
                traj = problem._march(time=scheme, x0=x0, span=span, substeps=sub)
            finally:
                problem._override = None
            return jnp.asarray(traj)[-1].reshape(-1)

        if not self.autonomous():  # its own times: compiled per window
            return jax.jit(run)([data[k] for k in keys], state)
        cache = problem.__dict__.setdefault("_cpl_march", {})
        key = (keys, span, sub, repr(scheme))
        if key not in cache:
            cache[key] = jax.jit(run)
        return cache[key]([data[k] for k in keys], state)

    def initial(self, problem):
        import jax.numpy as jnp

        if getattr(problem, "_time_order", 1) == 2:
            raise NotImplementedError("coupling: a u.tt problem marches [u; v]; its windows are not coupled yet.")
        return jnp.asarray(problem._initial_values(problem._params_scope())).reshape(-1)

    def _cfl_substeps(self, problem, state, time):
        """Substeps per step for the stable explicit step at ``state`` (the problem's CFL estimate)."""
        from ..utils.solver.time_route import _infer_time_window

        block = self.__dict__.get("_cfl_block")
        if block is None:
            t0, t1, dt = _infer_time_window(problem.domain)
            block = problem._march_parts(None, None, None, time, t0, t1, dt)[0]
            self._cfl_block = block
        limit = float(block.metadata["cfl_dt"](state, time.cfl))
        return max(1, int(np.ceil(float(block.dt) / limit - 1e-12)))

    def _face_values(self, problem, state, expr, faces, data):
        import jax.numpy as jnp

        g = problem.graph
        q = problem._split_dofs(state)
        qb = [qk[problem.ops.b_owner] for qk in q]
        ev = problem._evaluator(q, qb, None, problem._params_scope(self._scope(problem, data)))
        for i in range(problem.ops.boundary_passes):
            ev.solve_boundary(first=i == 0)
        if g.centering == "vertex":
            # A pinned (Dirichlet) vertex carries no normal derivative of its own: its row is the pin, so the
            # boundary rule never solves one. Read it from the vertex's least-squares gradient, exact for linear
            # fields (a conservative reaction would be better; see the commit).
            for k, cells, _ in problem._pins:
                Gk = problem.ops.cell_gradient(ev.q[k], ev.qb[k])
                gnk = jnp.einsum("fcd,fd->fc", Gk[problem.ops.b_owner], problem.ops.bn)
                mask = jnp.asarray(np.isin(np.asarray(g.b_owner), np.asarray(cells)))[:, None]
                ev.gn[k] = jnp.where(mask, gnk, ev.gn[k])
        return jnp.asarray(ev.rows_of(ev.values(expr, ("bfaces", faces)), ("bfaces", faces)))[:, 0]  # (n_faces,)

    def _attachment(self, problem, faces, iface):
        """``a_fi``: how much of boundary face ``f`` belongs to interface node ``i``."""
        import scipy.sparse as sp

        key = ("attach", tuple(faces[:4]), faces.size, id(iface))
        cache = self.__dict__.setdefault("_attach", {})
        if key in cache:
            return cache[key]
        g = problem.graph
        local = {int(n): i for i, n in enumerate(iface.nodes)}
        rows, cols, vals = [], [], []
        if g.centering == "vertex":  # a half-face belongs to its vertex
            from .transfer import _measure_fn

            mu = _measure_fn(iface.measure)(np.asarray(g.b_centroid[faces], float))
            for j, f in enumerate(faces):
                rows.append(local[int(g.b_owner[f])])
                cols.append(j)
                vals.append(float(np.linalg.norm(g.b_area[f])) * float(mu[j]))
        else:  # a facet's weight splits over its nodes as ∫_f φ_i dμ
            for j, f in enumerate(faces):
                nodes = [local[int(n)] for n in g.b_nodes[f]]
                sub = Interface.from_facets(
                    np.asarray(nodes), iface.points[nodes], np.arange(len(nodes))[None, :], iface.measure
                )
                for a, w in zip(nodes, sub.weights):
                    rows.append(a)
                    cols.append(j)
                    vals.append(w)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(iface.points.shape[0], faces.size)).toarray()
        cache[key] = A
        return A


class FEMParticipant(Participant):
    kind = "fem"

    def owns(self, node) -> bool:
        from ..trace import TrialFunction

        n = _unwrap(node)
        if not isinstance(n, TrialFunction):
            return False
        keys = getattr(self.problem, "_block_field_keys", None) or getattr(self.problem, "_trial_field_keys", None)
        return keys is not None and getattr(n, "field_key", None) in keys

    def constraints(self):
        return list(self.problem._constraints or [])

    def essential(self, problem=None) -> bool:
        return any(str(c).startswith("dirichlet") for c in (problem or self.problem).classification)

    def transient(self, problem=None) -> bool:
        return (problem or self.problem)._mode == "transient"

    def rebuild(self, extra):
        import jno

        return jno.fem(self.constraints() + list(extra), **(getattr(self.problem, "_fem_kwargs", None) or {}))

    def data(self, name):
        import jno

        node = jno.np.parameter(self.domain.fem_symbols()[0], name=name)  # a P1 field, valued at solve time
        node.model._runtime_field = True  # an essential value reads it per solve too, not frozen at build

        def setter(values, problem=None):
            return {name: np.asarray(values)}

        return node, setter

    def solve(self, problem, data, x0=None):
        import jax.numpy as jnp

        out = problem.solve(**data)
        out = out.fn() if hasattr(out, "fn") else out
        return jnp.asarray(out).reshape(-1)

    def march(self, problem, state, t0, window, data, time=None):
        """One window of the FEM's own time block: its step solvers (a direct step, composed once per scheme), a
        window copy of the block from ``state``, and the data as the runtime arguments. Going through
        ``fem.solve`` composed new step solvers on every call, a new compiled-march key each time (0.5 s a window,
        90 % of a contact coupling)."""
        import jax.numpy as jnp

        from .. import solve as _solve
        from ..utils.solver.backend_blocks import _block_time_grid, _default_transient_integrate, window_block
        from ..utils.solver.solver_api import compose_transient_step_solvers

        block = problem._op
        cache = problem.__dict__.setdefault("_cpl_steppers", {})
        key = repr(time)
        if key not in cache:
            cache[key] = compose_transient_step_solvers(None, _solve.lu(), None, problem, block, time)
        lin_s, nonlin_s = cache[key]
        win = window_block(block, self._span(t0, window), jnp.asarray(state).reshape(-1), who="coupling")
        args = {k: jnp.asarray(v) for k, v in data.items()}
        grid = _block_time_grid(win)
        if time is not None:
            ys = time.integrate(win, args, grid, linear_solve=lin_s, nonlinear_solve=nonlin_s)
        else:
            ys = _default_transient_integrate(win, args, grid, linear_solve=lin_s, nonlinear_solve=nonlin_s)
        return jnp.asarray(ys)[-1].reshape(-1)

    def initial(self, problem):
        import jax.numpy as jnp

        return jnp.asarray(problem._op.state0).reshape(-1)

    def _field_nodal(self, problem, state, field):
        """A scalar P1 field's nodal values from the DOF vector."""
        sl = problem.blocks[problem.block_index(field)]
        vals = np.asarray(state).reshape(-1)[sl.start : sl.stop]
        if vals.size != self.points.shape[0]:
            raise NotImplementedError(
                "coupling: an interface reads a jno.fem field as P1 nodal values; this field has "
                f"{vals.size} DOFs on a mesh of {self.points.shape[0]} nodes (higher order or vector)."
            )
        return vals

    def weak_terms(self, problem):
        """The problem's own weak terms (those with a test function): their free residual at a state is the
        consistent reaction, whose interface rows are the natural flux ``∫ N φ_i``."""
        from ..trace import TestFunction

        def has_test(n):
            n = _unwrap(n)
            return isinstance(n, TestFunction) or any(has_test(c) for c in _iter(n))

        return [c for c in self.constraints() if has_test(c)]

    def sample(self, problem, state, expr, tag, iface, data=None):
        import jax.numpy as jnp

        if has_derivative(expr, self.owns):
            # the natural flux, read as the consistent reaction ∫ N φ_i over the interface, per unit weight
            terms = self.weak_terms(problem)
            total = terms[0]
            for c in terms[1:]:
                total = total + c
            r = np.asarray(problem.eval(total, jnp.asarray(state)))
            field = self._flux_field(expr)
            sl = problem.blocks[problem.block_index(field)]
            # loads ∫ N φ_i -> the density whose loads they are (the interface mass, not the lumped weights,
            # so a linear flux comes back exactly)
            return jnp.asarray(iface.project(r[sl.start : sl.stop][iface.nodes]))

        def leaf(n):
            if self.owns(n):
                return self._field_nodal(problem, state, _unwrap(n))[iface.nodes]
            return None

        return jnp.asarray(evaluate_at(expr, iface.points, leaf, domain=self.domain)).reshape(-1)

    def _flux_field(self, expr):
        found = []

        def walk(n):
            n = _unwrap(n)
            if self.owns(n):
                found.append(n)
            for c in _iter(n):
                walk(c)

        walk(expr)
        keys = {getattr(f, "field_key", None) for f in found}
        if len(keys) != 1:
            raise NotImplementedError("coupling: a jno.fem flux on an interface must be the flux of one field.")
        return found[0]


def participant(problem) -> Participant:
    """The adapter for a ``jno.fem``, ``jno.fdm`` or ``jno.fvm`` problem."""
    from .._fem import FEM
    from .._fvm.trace import _TraceFVM
    from ..fdm import _TraceFDM

    if isinstance(problem, FEM):
        return FEMParticipant(problem)
    if isinstance(problem, _TraceFVM):
        return FVMParticipant(problem)
    if isinstance(problem, _TraceFDM):
        return FDMParticipant(problem)
    raise TypeError(f"coupling: {type(problem).__name__} is not a jno.fem, jno.fdm or jno.fvm problem.")
