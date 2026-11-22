"""Partitioned coupling of problems on their own meshes: interface conditions, who receives which, and the
iteration that makes them hold.

Based on:
    Giles, M. B. (1997). Stability analysis of numerical interface conditions in fluid-structure thermal
    analysis. International Journal for Numerical Methods in Fluids 25, 421-436 (the Dirichlet side is the
    one with the smaller interface stiffness).
    Küttler, U. & Wall, W. A. (2008). Fixed-point fluid-structure interaction solvers with dynamic relaxation.
    Computational Mechanics 43, 61-72 (Aitken relaxation).
    Quarteroni, A. & Valli, A. (1999). Domain Decomposition Methods for Partial Differential Equations.
    Oxford University Press (Dirichlet-Neumann and Robin-Robin iterations).

An interface condition is a residual that mixes the unknowns of two problems, written as for one problem:
``Tg - Ts`` (a value: the temperatures agree), ``kg*Tg.d(ng) + ks*Ts.d(ns)`` (a flux: what leaves one side
enters the other). Every condition is handed to ONE participant, which imposes it as its own boundary
condition with the other side's part replaced by data: the other participant evaluates that part at its own
interface nodes (a value directly, a flux from its own discrete flux) and it crosses the interface by the
consistent map (a value) or the conservative one (a flux). Which participant receives a condition follows
from its structure:

- a **value** and a **flux** condition between the same two participants are a Dirichlet-Neumann pair: the
  value goes to the side with the smaller interface stiffness (how much its flux changes for a unit change of
  its interface value, probed once), the flux to the other (Giles 1997);
- any other condition goes to the participant whose unknown it differentiates (a load on that side).

The participants then run in order, each with the data its senders produced last, until the data stop
changing. Data that flow against the order are relaxed (Aitken by default).
"""

from __future__ import annotations

import itertools
import operator
from dataclasses import dataclass, field

import numpy as np

from .._strong_form import _iter, _unwrap
from .participant import has_derivative, mentions, participant
from .transfer import conservative, consistent

_NAMES = itertools.count()  # data-field names, unique across every coupled problem in the process
_OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv, "**": operator.pow}


@dataclass(frozen=True)
class CouplingSpec:
    """How a coupled solve runs (see :func:`jno.solve.coupling`)."""

    scheme: str = "gauss-seidel"
    tol: float = 1e-8
    max_iter: int = 50
    accel: object = "aitken"
    dirichlet: object = None
    alpha: object = None
    measure: object = None
    omega0: float = 0.5

    def __post_init__(self):
        if self.scheme not in ("gauss-seidel", "jacobi"):
            raise ValueError(f"jno.solve.coupling(scheme={self.scheme!r}): use 'gauss-seidel' or 'jacobi'.")
        if not (self.alpha is None or self.alpha == "auto" or (isinstance(self.alpha, (int, float)) and self.alpha > 0)):
            raise ValueError(f"jno.solve.coupling(alpha={self.alpha!r}): use a positive number or 'auto'.")
        if not (self.accel is None or self.accel == "aitken" or isinstance(self.accel, (int, float))):
            raise ValueError(
                f"jno.solve.coupling(accel={self.accel!r}): use 'aitken', None, or a number (a fixed relaxation)."
            )


@dataclass
class _Data:
    """One data field: what ``sender`` evaluates (``expr`` at its ``sender_tag``) and ``receiver`` reads."""

    receiver: int
    sender: int
    expr: object
    flux: bool
    receiver_tag: str
    sender_tag: str
    node: object = None
    setter: object = None
    values: np.ndarray = None  # at the receiver's interface nodes


@dataclass
class _Condition:
    expr: object
    owners: list
    flux: bool
    tags: dict = field(default_factory=dict)  # participant -> its boundary tag in this condition
    coords: dict = field(default_factory=dict)  # participant -> its coordinate binding {name: Variable}
    receiver: int = None  # fixed by construction (a Robin condition), else decided by structure


def _spatial_tags(node):
    """The boundary tags of the spatial coordinate variables in ``node`` (normals excluded)."""
    from .._fvm.trace import _is_normal
    from .._strong_form import _iter_variables

    return {
        v.tag
        for v in _iter_variables(node)
        if getattr(v, "axis", "spatial") == "spatial" and not _is_normal(v) and v.tag is not None
    }


class Coupled:
    """A coupled problem: participants, their interface conditions, and the iteration."""

    def __init__(self, problems, conditions, spec: CouplingSpec | None = None):
        self.spec = spec or CouplingSpec()
        self.parts = [participant(p) for p in problems]
        self._all_data = []
        if len(self.parts) < 2:
            raise ValueError("jno.core([...]): a coupling needs at least two problems.")
        self.conditions = [self._analyse(c) for c in conditions]
        if not self.conditions:
            raise ValueError("jno.core([...]): problems on their own meshes need interface conditions between them.")
        self.stiffness = {}
        if self.spec.alpha is not None:
            self._robin()
        self.assign = self._assign()  # condition index -> receiving participant
        self._unfloat()
        self.order = self._order()
        self.data, self.problems = self._build()
        self.history = []
        self._last = {}

    # -- structure ------------------------------------------------------------------------------------
    def _who(self, node):
        return [i for i, p in enumerate(self.parts) if mentions(node, p.owns)]

    def _analyse(self, expr):
        owners = self._who(expr)
        if len(owners) < 2:
            raise ValueError(
                "jno.core([...]): an interface condition must mix the unknowns of two problems; this one reads "
                f"{'only one' if owners else 'none'}. Put a single problem's condition in that problem."
            )
        if len(owners) > 2:
            raise NotImplementedError("jno.core([...]): an interface condition between three problems at once.")
        cond = _Condition(expr, owners, any(has_derivative(expr, self.parts[i].owns) for i in owners))
        # each side's binding: the views keep every binding they combine (`_coord_sets`), and a binding's
        # Variables know their domain, so each side reads its own region whatever the tags are called
        sets = getattr(expr, "_coord_sets", None) or ((getattr(expr, "_coord_vars", None) or {}),)
        for i in owners:
            dom = self.parts[i].domain
            mine = [
                cv
                for cv in sets
                if any(
                    getattr(v, "_domain", None) is dom for v in cv.values() if getattr(v, "axis", "spatial") == "spatial"
                )
            ]
            tags = {v.tag for cv in mine for v in cv.values() if getattr(v, "axis", "spatial") == "spatial"}
            if len(tags) != 1:
                raise ValueError(
                    f"jno.core([...]): in an interface condition, the {self.parts[i].kind} problem's part must "
                    f"be bound to one boundary tag of its own mesh (`u.bind(x=xb, y=yb)`); it is bound to "
                    f"{sorted(tags) or 'none'}."
                )
            cond.tags[i] = tags.pop()
            cv = {}
            for c in mine:
                cv.update(c)
            cond.coords[i] = cv
        return cond

    def _owned_subtrees(self, expr, i):
        """The maximal subtrees of ``expr`` that read participant ``i``'s unknowns and no other's."""
        out = []

        def walk(node):
            who = self._who(node)
            if who == [i]:
                out.append(node)
                return
            if i in who:
                for c in _iter(_unwrap(node)):
                    walk(c)

        walk(expr)
        return out

    def _pairs(self):
        """``{(a, b): (value condition, flux condition)}`` for every Dirichlet-Neumann pair not yet assigned."""
        pairs = {}
        for k, c in enumerate(self.conditions):
            if c.receiver is None:
                pairs.setdefault(tuple(c.owners), []).append(k)
        out = {}
        for ab, ks in pairs.items():
            values = [k for k in ks if not self.conditions[k].flux]
            fluxes = [k for k in ks if self.conditions[k].flux]
            if len(values) == 1 and len(fluxes) == 1:
                out[ab] = (values[0], fluxes[0])
        return out

    def _robin(self, only=None, alpha=None):
        """Value/flux pairs as two Robin conditions, ``F + α·σ_p·V`` for each side ``p`` (``σ_p`` the sign that
        makes its own value term positive), so both sides read both of the other's quantities (Robin-Robin,
        Quarteroni & Valli 1999). ``alpha="auto"`` takes ``√(S_a S_b)`` from the stiffness probe. ``only``
        limits it to those pairs."""
        from .._fvm.trace import _signed_terms

        out, used = [], set()
        for (a, b), (kv, kf) in self._pairs().items():
            if only is not None and (a, b) not in only:
                continue
            V, F = self.conditions[kv].expr, self.conditions[kf].expr
            al = self.spec.alpha if alpha is None else alpha
            if al == "auto":
                al = float(np.sqrt(max(self._stiffness(a, kv, kf), 1e-300) * max(self._stiffness(b, kv, kf), 1e-300)))
            self.alpha = al
            alpha_ab = al
            for p in (a, b):
                own = [sg for sg, t in _signed_terms(V, distribute=False) if self._who(t) == [p]]
                if len(own) != 1:
                    raise ValueError(
                        "jno.core([...]): Robin-Robin needs each side's value as one term of the value condition."
                    )
                cond = self._analyse(F + (alpha_ab * own[0]) * V)
                cond.receiver = p
                out.append(cond)
            used |= {kv, kf}
        self.conditions = [c for k, c in enumerate(self.conditions) if k not in used] + out

    def _assign(self):
        """Condition -> receiver. A value/flux pair between two participants is Dirichlet-Neumann; any other
        condition goes to the participant whose unknown it differentiates."""
        assign = {}
        pairs = {}
        for k, c in enumerate(self.conditions):
            pairs.setdefault(tuple(c.owners), []).append(k)
        for k, c in enumerate(self.conditions):
            if c.receiver is not None:
                assign[k] = c.receiver
        for (a, b), ks in pairs.items():
            ks = [k for k in ks if self.conditions[k].receiver is None]
            if not ks:
                continue
            values = [k for k in ks if not self.conditions[k].flux]
            fluxes = [k for k in ks if self.conditions[k].flux]
            if len(values) > 1 or len(fluxes) > 1:
                raise NotImplementedError(
                    "jno.core([...]): between two problems, one value condition and one flux condition per "
                    "interface (several coupled fields are not supported yet)."
                )
            if values and fluxes:
                d = self._dirichlet_side(a, b, values[0], fluxes[0])
                assign[values[0]] = d
                assign[fluxes[0]] = b if d == a else a
            elif fluxes:
                k = fluxes[0]
                diff = [i for i in (a, b) if has_derivative(self.conditions[k].expr, self.parts[i].owns)]
                if len(diff) != 1:
                    raise ValueError(
                        "jno.core([...]): a flux condition without a value condition is a load on one side; this "
                        "one differentiates both. Add the value condition (the fields agree) to make a pair."
                    )
                assign[k] = diff[0]
            else:
                raise ValueError(
                    "jno.core([...]): a value condition alone does not couple two problems; add the flux "
                    "condition (what leaves one side enters the other)."
                )
        return assign

    def _dirichlet_side(self, a, b, kv, kf):
        """The participant that receives the value condition of a Dirichlet-Neumann pair."""
        given = self.spec.dirichlet
        if given is not None:
            for i in (a, b):
                if given is self.parts[i].problem:
                    return i
            raise ValueError("jno.solve.coupling(dirichlet=...): pass one of the coupled problems.")
        s = {i: self._stiffness(i, kv, kf) for i in (a, b)}
        return a if s[a] <= s[b] else b

    def _stiffness(self, i, kv, kf):
        """How much participant ``i``'s interface flux changes for a unit rise of its interface value, with the
        values of its OTHER interfaces held (the diagonal of its Steklov-Poincaré operator): solved with every
        value condition it takes part in as its Dirichlet condition, this one's data at 0 and at 1 (a linear
        probe; the mean over the interface, per unit weight). Held open, another interface would let a floating
        participant follow the probe and read no stiffness at all."""
        key = (i, kv)
        if key in self.stiffness:
            return self.stiffness[key]
        part = self.parts[i]
        cf = self.conditions[kf]
        mine = [k for k, c in enumerate(self.conditions) if not c.flux and i in c.owners and c.receiver is None]
        probe = self.__dict__.setdefault("_probe", {})
        if i not in probe:
            received = [(k, *self._receive(i, self.conditions[k])) for k in mine]
            probe[i] = (part.rebuild([c for _, c, _ in received]), {k: d for k, _, d in received})
        prob, data = probe[i]
        iface = part.interface(self.conditions[kv].tags[i], self.spec.measure)
        flux = self._own_part(cf.expr, i)
        out = []
        for level in (0.0, 1.0):
            extra = {}
            for k, ds in data.items():
                for d in ds:
                    face = part.interface(self.conditions[k].tags[i], self.spec.measure)
                    lv = level if k == kv else 0.0
                    extra.update(d.setter(part.nodal(face, np.full(face.points.shape[0], lv)), prob))
            state = part.solve(prob, extra)
            out.append(np.asarray(part.sample(prob, state, flux, cf.tags[i], iface, extra)))
        self.stiffness[key] = abs(float(iface.weights @ (out[1] - out[0]))) / iface.total
        return self.stiffness[key]

    def _unfloat(self):
        """A steady participant with no essential condition of its own that takes only fluxes is singular (a
        floating subdomain). One of its pairs becomes Robin-Robin, ``α = √(S_i S_j)``: the ``α·value`` term pins
        it. The pair chosen is the one whose two stiffnesses are closest, where Dirichlet-Neumann is weakest."""
        for i, part in enumerate(self.parts):
            if part.transient() or part.essential():
                continue
            if any(self.assign[k] == i and not c.flux for k, c in enumerate(self.conditions)):
                continue
            if any(self.assign[k] == i and c.receiver is not None for k, c in enumerate(self.conditions)):
                continue  # already a Robin condition
            pairs = {ab: kk for ab, kk in self._pairs().items() if i in ab}
            if not pairs:
                continue

            def balance(item):
                (a, b), (kv, kf) = item
                sa, sb = self._stiffness(a, kv, kf), self._stiffness(b, kv, kf)
                return max(sa, sb) / max(min(sa, sb), 1e-300)

            (a, b), (kv, kf) = min(pairs.items(), key=balance)
            alpha = float(np.sqrt(max(self._stiffness(a, kv, kf), 1e-300) * max(self._stiffness(b, kv, kf), 1e-300)))
            self._robin(only={(a, b)}, alpha=alpha)
            self.assign = self._assign()

    def _own_part(self, expr, i):
        """The part of a flux condition that is participant ``i``'s own flux (its terms that differentiate it)."""
        from .._fvm.trace import _signed_terms

        terms = [
            (s, t)
            for s, t in _signed_terms(expr, distribute=False)
            if mentions(t, self.parts[i].owns) and not any(mentions(t, p.owns) for j, p in enumerate(self.parts) if j != i)
        ]
        if not terms:
            raise ValueError("jno.core([...]): no term of this flux condition is the problem's own flux.")
        total = None
        for s, t in terms:
            term = t if s > 0 else -1.0 * t
            total = term if total is None else total + term
        return total

    def _receive(self, i, cond):
        """Condition ``cond`` as participant ``i``'s own: the other participant's maximal subtrees replaced by
        data fields on ``i``'s mesh. Returns the rewritten condition and its new :class:`_Data`."""
        part = self.parts[i]
        data = []

        def rewrite(node):
            n = _unwrap(node)
            who = self._who(node)
            if not who:
                return node
            if who == [i]:
                return node
            if i not in who:
                (q,) = who
                d = _Data(
                    receiver=i,
                    sender=q,
                    expr=node,
                    flux=has_derivative(node, self.parts[q].owns),
                    receiver_tag=cond.tags[i],
                    sender_tag=cond.tags[q],
                )
                d.node, d.setter = part.data(f"cpl_{next(_NAMES)}")
                data.append(d)
                return d.node
            from ..trace import BinaryOp, FunctionCall

            if isinstance(n, BinaryOp):
                return _OPS[n.op](rewrite(n.left), rewrite(n.right))
            if isinstance(n, FunctionCall):
                return n.copy_with_args([rewrite(a) for a in n.args])
            raise NotImplementedError(
                f"jno.core([...]): an interface condition combines the two problems inside a {type(n).__name__}; "
                "write it as sums and products of each problem's own expressions."
            )

        out = rewrite(cond.expr)
        try:  # the receiver's own region, which its classifier reads from the binding
            out._coord_vars = dict(cond.coords[i])
        except AttributeError:
            pass
        return out, data

    def _order(self):
        """Gauss-Seidel order: a Dirichlet side before its Neumann side, a load's sender before its receiver."""
        n = len(self.parts)
        after = {i: set() for i in range(n)}  # i must come after these
        for k, c in enumerate(self.conditions):
            r = self.assign[k]
            (s,) = [j for j in c.owners if j != r]
            if c.flux:  # the flux receiver runs after the side that produces the flux
                after[r].add(s)
        order, placed = [], set()
        while len(order) < n:
            ready = [i for i in range(n) if i not in placed and after[i] <= placed]
            if not ready:
                ready = [min(i for i in range(n) if i not in placed)]  # a cycle: break it at the first
            order.append(ready[0])
            placed.add(ready[0])
        return order

    def _build(self):
        """Every participant's problem rebuilt with the conditions it receives, and the data fields they read."""
        self._all_data = []
        extra = {i: [] for i in range(len(self.parts))}
        for k, c in enumerate(self.conditions):
            r = self.assign[k]
            cond, data = self._receive(r, c)
            self._all_data.extend(data)
            extra[r].append(self._as_condition(r, cond, c))
        problems = [p.rebuild(extra[i]) if extra[i] else p.problem for i, p in enumerate(self.parts)]
        for d in self._all_data:
            iface = self.parts[d.receiver].interface(d.receiver_tag, self.spec.measure)
            d.values = np.zeros(iface.points.shape[0])
        return self._all_data, problems

    def _as_condition(self, i, cond, original):
        """The received condition in the participant's own form. A strong-form problem takes it as written; a
        finite-element problem takes a flux condition as a natural boundary term: ``F + R = 0`` with ``F`` its
        own (natural) flux becomes the weak term ``R·φ`` on that boundary."""
        part = self.parts[i]
        if part.kind != "fem" or not original.flux:
            return cond
        from .._fvm.trace import _signed_terms

        rest = None
        for s, t in _signed_terms(cond, distribute=False):
            if has_derivative(t, part.owns):
                continue  # the natural flux, which the weak form already carries
            term = t if s > 0 else -1.0 * t
            rest = term if rest is None else rest + term
        if rest is None:
            raise ValueError("jno.core([...]): the flux condition leaves nothing for the finite-element side to impose.")
        return rest * self._test_function(i, original)

    def _test_function(self, i, original):
        """The test function of the field this condition is about, bound to the interface variables."""
        from ..trace import TestFunction

        part = self.parts[i]
        field = part._flux_field(self._own_part(original.expr, i))
        found = []

        def walk(n):
            n = _unwrap(n)
            if isinstance(n, TestFunction) and getattr(n, "field_key", None) == field.field_key:
                found.append(n)
            for c in _iter(n):
                walk(c)

        for c in part.constraints():
            walk(c)
        if not found:
            raise ValueError("jno.core([...]): the finite-element problem has no test function for this field.")
        return found[0].bind(**original.coords[i])

    # -- iteration ---------------------------------------------------------------------------------------
    def _produce(self, d, states):
        """Data ``d`` from its sender's current state, carried to the receiver's interface nodes."""
        s, r = self.parts[d.sender], self.parts[d.receiver]
        src = s.interface(d.sender_tag, self.spec.measure)
        dst = r.interface(d.receiver_tag, self.spec.measure)
        vals = np.asarray(
            s.sample(self.problems[d.sender], states[d.sender], d.expr, d.sender_tag, src, self._last.get(d.sender))
        )
        M = self._map(d, dst, src)
        return np.asarray(M @ vals)

    def _map(self, d, dst, src):
        cache = self.__dict__.setdefault("_maps", {})
        key = (d.receiver, d.receiver_tag, d.sender, d.sender_tag, d.flux)
        if key not in cache:
            cache[key] = (conservative if d.flux else consistent)(dst, src)
        return cache[key]

    def _solve_part(self, i, states):
        part, prob = self.parts[i], self.problems[i]
        extra = {}
        for d in self.data:
            if d.receiver == i:
                iface = part.interface(d.receiver_tag, self.spec.measure)
                extra.update(d.setter(part.nodal(iface, d.values), prob))
        self._last[i] = extra  # what its boundary conditions read: its own samples need them too
        return part.solve(prob, extra, x0=states[i])

    def solve(self):
        """Iterate to the coupled solution: a list with each problem's solution, in the order given."""
        n = len(self.parts)
        states = [None] * n
        pos = {i: k for k, i in enumerate(self.order)}
        back = [d for d in self.data if pos[d.sender] >= pos[d.receiver]]  # read before their sender runs
        if self.spec.scheme == "jacobi":
            back = list(self.data)
        x = np.concatenate([d.values for d in back]) if back else np.zeros(0)
        omega, r_prev = None, None
        self.history = []
        for it in range(1, self.spec.max_iter + 1):
            self._set(back, x)
            if self.spec.scheme == "jacobi":
                states = [self._solve_part(i, states) for i in range(n)]
            else:
                for i in self.order:
                    for d in self.data:  # forward data: fresh from senders already run in this sweep
                        if d.receiver == i and d not in back:
                            d.values = self._produce(d, states)
                    states[i] = self._solve_part(i, states)
            x_new = np.concatenate([self._produce(d, states) for d in back]) if back else np.zeros(0)
            r = x_new - x
            scale = max(float(np.linalg.norm(x_new)), 1e-300)
            err = float(np.linalg.norm(r)) / scale
            self.history.append(err)
            if err <= self.spec.tol or not back:
                self._set(back, x_new)
                break
            h = self.history
            if len(h) >= 4 and max(h[-3:]) <= 1.5 * min(h[-3:]) and h[-1] < 1e-4:
                raise RuntimeError(
                    f"jno.core([...]): the coupling stalled at an interface change of {err:.1e} (history "
                    f"{self._brief()}), above tol = {self.spec.tol:.1e}: the participants' own solves are not that "
                    "accurate. Set tol above it, or solve the participants more accurately (a direct solver)."
                )
            omega = self._relaxation(omega, r, r_prev)
            x = x + omega * r
            r_prev = r
        else:
            raise RuntimeError(
                f"jno.core([...]): the coupling did not converge in {self.spec.max_iter} iterations (interface "
                f"change {self.history[-1]:.2e} > tol {self.spec.tol:.1e}; history {self._brief()}). Raise max_iter, "
                "relax (accel='aitken' or a number below 1), or swap the Dirichlet side (dirichlet=...)."
            )
        self.iterations = len(self.history)
        self.states = states
        return [self._shape(i, st) for i, st in enumerate(states)]

    def _brief(self):
        h = self.history
        return ", ".join(f"{e:.1e}" for e in (h[:3] + ["…"] + h[-2:] if len(h) > 6 else h) if not isinstance(e, str))

    def _set(self, back, x):
        k = 0
        for d in back:
            m = d.values.size
            d.values = np.asarray(x[k : k + m])
            k += m

    def _relaxation(self, omega, r, r_prev):
        a = self.spec.accel
        if a is None:
            return 1.0
        if isinstance(a, (int, float)):
            return float(a)
        if r_prev is None or omega is None:  # Aitken (Küttler & Wall 2008)
            return self.spec.omega0
        dr = r - r_prev
        denom = float(dr @ dr)
        if denom <= 0.0:
            return omega
        return float(-omega * (r_prev @ dr) / denom)

    def _shape(self, i, state):
        p = self.problems[i]
        part = self.parts[i]
        if part.kind == "fem":
            return np.asarray(state)
        nf, N = getattr(p, "_nf", 1), getattr(p, "_N", None)
        return np.asarray(state).reshape(N) if nf == 1 else np.asarray(state).reshape(nf, N)


def detect(constraints):
    """``(problems, conditions)`` when a ``jno.core`` list holds two or more problems on their own meshes (no
    shared-mesh region; those are ``jno.dd``'s) and, besides them, only conditions; else ``None``."""
    from .._fem import FEM
    from .._fvm.trace import _TraceFVM
    from ..fdm import _TraceFDM

    kinds = (FEM, _TraceFVM, _TraceFDM)
    problems = [c for c in constraints if isinstance(c, kinds)]
    if len(problems) < 2 or any(getattr(p, "region_geometry", None) is not None for p in problems):
        return None
    return problems, [c for c in constraints if not isinstance(c, kinds)]


def couple(problems, conditions, spec=None):
    """The coupled problem of ``problems`` (each on its own mesh) joined by the interface ``conditions``."""
    return Coupled(problems, conditions, spec)
