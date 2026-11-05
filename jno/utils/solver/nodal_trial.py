"""A neural operator's output, declared a field on the mesh, written where the unknown goes in ``jno.fem``.

An operator network -- an FNO on the mesh's grid, a DeepONet at the nodes -- predicts the FE coefficients
of the solution for an input field. ``.field.bind(x=x, y=y)`` declares its output a field on the mesh
(the coordinates are not network inputs), and in the weak form it is written like the FE trial::

    u = net(rho).field.bind(x=xi, y=yi)            # rho: a per-sample field of a batched S * domain
    pde = jno.fem([u.x * vi.x + u.y * vi.y - rho * vi, U(xb, yb) - 0.0])
    jno.core([pde.solve(precond=jno.precond.gmg()).mse])     # P (A u - b(rho)), per sample

``jno.fem`` assembles the form once with the FE trial in the field's place -- so ``u.x`` is the gradient
of the FE function with the network's output as nodal values -- and the input fields as nodal field
parameters, and returns a :class:`NetworkTrialResidual`: the residual ``A u - b(rho)`` of the prediction,
per sample. The essential rows are imposed, not trained for: the residual is zero there and
:attr:`NetworkTrialResidual.field` carries the prescribed values. A network of the coordinates
(``net(x, y)``) is not a field and stays a VPINN trial.

Grid outputs are read in the ``"ij"`` convention -- ``out[ix, iy(, iz)]``, x first -- on any
tensor-product mesh; an output with one entry per node is read in the FE node order.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np

from ...trace import Hessian, Jacobian, ModelCall, Placeholder, TensorTag, TestFunction, TrialFunction, Variable, substitute
from .parametric_helpers import _is_runtime_scalar_parameter


def _walk(node):
    from ..._fem import _walk as walk

    return walk(node)


def _bare(obj):
    from ..._fem import _bare as bare

    return bare(obj)


def _has(node, cls) -> bool:
    return any(isinstance(n, cls) for n in _walk(node))


def _is_network(n) -> bool:
    return isinstance(n, ModelCall) and not _is_runtime_scalar_parameter(n) and bool(n.args)


def _is_fd_partial(n) -> bool:
    return isinstance(n, (Jacobian, Hessian)) and str(getattr(n, "scheme", "")).startswith("finite_difference")


def split_network_trial(constraints) -> Optional[Dict[str, Any]]:
    """Recognise a form whose trial is a declared field (``net(rho).field``); ``None`` for every other form.

    Returns the rewritten constraints (the field replaced by the FE trial, its finite-difference partials
    by the trial's, each input data field by a nodal field parameter) and what is needed to evaluate the
    residual per sample. A network that depends on no coordinate and is not declared a field is refused:
    as a VPINN trial its derivatives would be zero.
    """
    items = list(constraints) if isinstance(constraints, (list, tuple)) else [constraints]
    bares = [_bare(c) for c in items]
    is_weak = [isinstance(b, Placeholder) and _has(b, TestFunction) for b in bares]
    weak = [b for b, w in zip(bares, is_weak) if w]
    if not weak or any(_has(b, TrialFunction) for b in weak):
        return None

    fields, partials = {}, []
    for b in weak:
        for n in _walk(b):
            if _is_fd_partial(n) and any(_is_network(m) for m in _walk(n.target)):
                if not _is_network(n.target):
                    raise NotImplementedError(
                        "jno.fem: declare the field on the network call itself, `net(rho).field.bind(...)`, "
                        "and build expressions from that field; a derived expression as the trial is not supported."
                    )
                fields[id(n.target)] = n.target
                partials.append(n)
            elif _is_network(n) and getattr(n, "_mesh_field", False):
                fields[id(n)] = n
    undeclared = [
        n
        for b in weak
        for n in _walk(b)
        if _is_network(n) and id(n) not in fields and not any(isinstance(m, Variable) for a in n.args for m in _walk(a))
    ]
    if undeclared:
        raise ValueError(
            f"jno.fem: {undeclared[0]!r} depends on no coordinate, so as a trial its derivatives would be 0. If "
            "it is a neural operator whose output is a field on the mesh, declare it: "
            "`u = net(rho).field.bind(x=xi, y=yi)`, then write the form with `u.x`, `u.y` like the FE trial."
        )
    if not fields:
        return None
    if len(fields) > 1:
        raise NotImplementedError(
            "jno.fem: the weak form has more than one network field; a single field trial is supported for now."
        )
    (net,) = fields.values()

    test = next(n for b in weak for n in _walk(b) if isinstance(n, TestFunction))
    if getattr(test, "value_shape", ()) not in ((), None):
        raise NotImplementedError("jno.fem: a network field trial is scalar only for now (value_shape=()).")
    trial = next((n for b in bares if isinstance(b, Placeholder) for n in _walk(b) if isinstance(n, TrialFunction)), None)
    if trial is None:  # no essential condition: the FE trial paired with the test function
        trial = TrialFunction(name="u", value_shape=test.value_shape, order=test.order, space=test.space)
        trial.field_key, trial._test, trial._domain = test.field_key, test, test._domain

    # The field's finite-difference partials become the FE trial's own (AD over its shape functions).
    mapping = {net: trial}
    for n in partials:
        mapping[n] = (
            Jacobian(trial, list(n.variables))
            if isinstance(n, Jacobian)
            else Hessian(trial, list(n.variables), trace=n.trace)
        )
    weak = [substitute(b, mapping) for b in weak]  # the network's own inputs leave the form with it

    from ...architectures.models import parameter

    data: Dict[str, Any] = {}
    mapping = {}
    for b in weak:
        for n in _walk(b):
            if isinstance(n, TensorTag) and n.tag not in data:
                data[n.tag] = n
                mapping[n] = _bare(parameter(test, name=n.tag))
    weak = iter([substitute(b, mapping) for b in weak] if mapping else weak)
    rewritten = [next(weak) if w else c for w, c in zip(is_weak, items)]
    return {"constraints": rewritten, "network": net, "data": data}


def _node_reader(points: np.ndarray):
    """``read(out) -> (n_nodes,)``: one sample's network output (a grid, or one value per node) at the nodes."""
    n, dim = points.shape
    axes = [np.unique(np.round(points[:, k], 12)) for k in range(dim)]
    grid = tuple(len(a) for a in axes)
    index = None
    if int(np.prod(grid)) == n:
        index = tuple(np.searchsorted(a, np.round(points[:, k], 12)) for k, a in enumerate(axes))
        if len(set(zip(*index))) != n:
            index = None

    def read(out):
        out = jnp.asarray(out)
        shape = tuple(s for s in out.shape if s != 1)
        if shape == (n,):
            return out.reshape(n)
        if index is not None and shape == grid:
            return out.reshape(grid)[index]
        expected = f"{n} values (one per node, in the FE node order)"
        if index is not None:
            expected += f" or a {grid} grid indexed [ix, iy{', iz' if dim == 3 else ''}] (x first)"
            if shape == grid[::-1]:
                expected += f"; got {shape}, the transposed grid -- swap the axes"
        raise ValueError(
            f"jno.fem (network field trial): a field of shape {tuple(out.shape)} does not fit this mesh; expected {expected}."
        )

    return read


class NetworkTrialResidual:
    """The physics-informed loss of a network field: ``jno.fem`` with ``u = net(rho).field.bind(...)`` as the trial.

    ``pde.solve()`` is the FE residual ``A u - b(rho)`` of the prediction, per sample, zero on the
    essential rows; ``pde.solve(precond=spec)`` is ``P (A u - b)`` with any ``jno.precond`` spec built
    from the assembled ``A`` (a multigrid cycle makes the loss track the error; Wen, Mishra & Zeinhofer,
    arXiv:2609.36216). Both are trace expressions for ``jno.core``; ``pde.mse`` is ``pde.solve().mse``.
    ``pde.field`` is the prediction in the FE node order with the essential values set, and ``pde.fem``
    the assembled problem behind it.
    """

    def __init__(self, fem, network, data):
        from ..._fem import _essential_dof_ids

        self.fem = fem
        self._network = network
        self._data = data
        self._read = _node_reader(np.asarray(fem.points))
        n = int(fem.dofs)
        mask = np.zeros(n)
        mask[np.asarray(_essential_dof_ids(fem.domain, fem._op), dtype=np.int64)] = 1.0
        self._essential = jnp.asarray(mask)
        self._residual_fn = fem.residual
        self._args_zero = {k: jnp.zeros(n) for k in data}
        self._A = None
        # An essential row of the assembled residual reads s_d (u_d - g_d), its scale s_d set by the
        # elimination; it depends on u_d alone, so one probe with u = 1 on those rows measures it.
        r0, r1 = (self._call(jnp.asarray(v), self._args_zero) for v in (np.zeros(n), mask))
        self._scale = jnp.where(self._essential > 0, r1 - r0, 1.0)

    def _call(self, u, args):
        return self._residual_fn(u, args) if args else self._residual_fn(u)

    def _residual(self, out, *fields):
        u = self._read(out)
        return u, self._call(u, {k: self._read(f) for k, f in zip(self._data, fields)})

    def _operator(self):
        """The assembled A, once; refused when the input fields change it (one P cannot serve every sample)."""
        if self._A is None:
            op = self.fem.operator
            if hasattr(op, "evaluate"):
                A0, _ = op.evaluate(self._args_zero)
                A1, _ = op.evaluate({k: jnp.ones_like(v) for k, v in self._args_zero.items()})
                z = jnp.asarray(np.random.default_rng(0).normal(size=int(self.fem.dofs)), A0.dtype)
                # Relative, at sqrt(eps): re-assembling A in float32 on a GPU moves it by round-off (~1e-7),
                # while an input field that really enters A changes it by O(1) between 0 and 1.
                a0, a1 = np.asarray(A0 @ z, np.float64), np.asarray(A1 @ z, np.float64)
                if np.linalg.norm(a1 - a0) > np.sqrt(np.finfo(np.dtype(A0.dtype)).eps) * np.linalg.norm(a0):
                    raise NotImplementedError(
                        f"jno.fem (network field trial): the input field(s) {sorted(self._data)} enter the operator A, "
                        "so A differs per sample and one preconditioner built for all of them would be wrong for "
                        "most. A per sample preconditioner is not supported yet; use the plain residual "
                        "(`pde.solve()`), or keep the input fields in the load."
                    )
                self._A = A0
            else:
                self._A = op[0]
        return self._A

    def solve(self, precond=None):
        """The loss term: ``A u - b`` per sample, or ``P (A u - b)`` with a ``jno.precond`` spec."""
        import jno

        keep = 1.0 - self._essential
        if precond is None:
            fn, name = (lambda out, *f: keep * self._residual(out, *f)[1]), "A u - b"
        else:
            from .solver_api import LinearOperator, PrecondContext, materialize_precond

            P = materialize_precond(precond, PrecondContext(LinearOperator(self._operator()), self.fem))
            fn, name = (lambda out, *f: keep * P(keep * self._residual(out, *f)[1])), "P (A u - b)"
        return jno.fn(fn, [self._network, *self._data.values()], name=name)

    @property
    def mse(self):
        return self.solve().mse

    @property
    def field(self):
        """The prediction at the nodes (FE order) with the essential values ``g`` set in place of the network's."""
        import jno

        def field(out, *f):
            u, r = self._residual(out, *f)
            return u - self._essential * r / self._scale

        return jno.fn(field, [self._network, *self._data.values()], name="u")

    def __repr__(self):
        return f"NetworkTrialResidual(dofs={self.fem.dofs}, inputs={sorted(self._data)})"
