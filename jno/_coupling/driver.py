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
from .participant import _tracing, has_derivative, mentions, participant
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
    implicit: bool = True
    window: object = None
    time: object = None
    reuse: int = 0
    x0: object = None
    compile: bool = True

    def __post_init__(self):
        if self.scheme not in ("gauss-seidel", "jacobi"):
            raise ValueError(f"jno.solve.coupling(scheme={self.scheme!r}): use 'gauss-seidel' or 'jacobi'.")
        if not (self.alpha is None or self.alpha == "auto" or (isinstance(self.alpha, (int, float)) and self.alpha > 0)):
            raise ValueError(f"jno.solve.coupling(alpha={self.alpha!r}): use a positive number or 'auto'.")
        if not (self.accel is None or self.accel in ("aitken", "iqn-ils") or isinstance(self.accel, (int, float))):
            raise ValueError(
                f"jno.solve.coupling(accel={self.accel!r}): use 'aitken', 'iqn-ils', None, or a number (a fixed "
                "relaxation)."
            )


@dataclass
class _Data:
    """One data field: what ``sender`` evaluates (``expr`` at its ``sender_tag``) and ``receiver`` reads. Its values
    are not kept here: the iteration passes them along, one array per field in the order of ``Coupled.data``."""

    receiver: int
    sender: int
    expr: object
    flux: bool
    receiver_tag: str
    sender_tag: str
    node: object = None
    setter: object = None
    cond: int = None  # the index of the condition it was made for
    received: object = None  # that condition as its receiver imposes it (this field's node in it)


@dataclass
class _Condition:
    expr: object
    owners: list
    flux: bool
    tags: dict = field(default_factory=dict)  # participant -> its boundary tag in this condition
    coords: dict = field(default_factory=dict)  # participant -> its coordinate binding {name: Variable}
    receiver: int = None  # fixed by construction (a Robin condition), else decided by structure


def _nodes(node):
    """Every node of an expression."""
    n = _unwrap(node)
    yield n
    for c in _iter(n):
        yield from _nodes(c)


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
        self._probe_params = self._initial_params()  # what the structural probes give the trainable parameters
        self._guess = self._guesses()  # participant -> the state a steady one starts from (x0=)
        transient = [i for i, part in enumerate(self.parts) if part.transient()]
        self._window = self._windows(transient) if transient else None  # (window, t0, t1)
        if self.spec.alpha is not None:
            self._robin()
        self.assign = self._assign()  # condition index -> receiving participant
        self._unfloat()
        self.order = self._order()
        self.data, self.problems = self._build()
        self.history = []
        self.compiles = 0  # traces of the compiled window program (a window of one sweep)

    # -- structure ------------------------------------------------------------------------------------
    def _who(self, node):
        """The participants ``node`` belongs to: whose unknowns it reads, and whose coordinates or normals (a
        Variable knows its domain), so ``p_g * n_s`` -- the gas pressure on the wall normal -- reads the gas for
        ``p_g`` and the wall for ``n_s``."""
        return [i for i, p in enumerate(self.parts) if mentions(node, p.owns) or self._reads_domain(node, p.domain)]

    @staticmethod
    def _reads_domain(node, domain):
        from .._strong_form import _iter_variables

        return any(getattr(v, "_domain", None) is domain for v in _iter_variables(node))

    def _analyse(self, expr):
        owners = [i for i, p in enumerate(self.parts) if mentions(expr, p.owns)]
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
            if not tags:  # no binding survived (git-bug 3c78460): its own Variables in the tree say where it is
                tags = self._tags_from_variables(expr, dom)
                if len(tags) == 1:
                    (tag,) = tags
                    xs = dom.variable(tag, split=True)
                    mine = [{n: v for n, v in zip(("x", "y", "z"), xs) if getattr(v, "axis", "spatial") == "spatial"}]
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

    @staticmethod
    def _tags_from_variables(expr, dom):
        """Boundary tags of ``dom``'s own Variables in ``expr``: coordinates by their tag, normals ``n_<tag>``."""
        from .._fvm.trace import _is_normal
        from .._strong_form import _iter_variables

        tags = set()
        for v in _iter_variables(expr):
            if getattr(v, "_domain", None) is not dom or getattr(v, "axis", "spatial") == "temporal":
                continue
            tag = str(v.tag)
            tags.add(tag[2:] if _is_normal(v) and tag.startswith("n_") else tag)
        return tags

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
            paired = set()
            for kv in values:  # a value pairs with the flux condition of the same field: Dirichlet-Neumann
                mine = self._fields(self.conditions[kv].expr, derivative=False)
                match = [kf for kf in fluxes if kf not in paired and self._fields(self.conditions[kf].expr, True) & mine]
                if len(match) != 1:
                    raise ValueError(
                        "jno.core([...]): a value condition couples two problems only with the flux condition of the "
                        f"same field (what leaves one side enters the other); found {len(match)} for one of them."
                    )
                kf = match[0]
                paired.add(kf)
                d = self._dirichlet_side(a, b, kv, kf)
                assign[kv] = d
                assign[kf] = b if d == a else a
            for k in fluxes:  # any other flux condition is a load on the side it differentiates
                if k in paired:
                    continue
                diff = [i for i in (a, b) if has_derivative(self.conditions[k].expr, self.parts[i].owns)]
                if len(diff) != 1:
                    raise ValueError(
                        "jno.core([...]): a flux condition without a value condition is a load on one side; this one "
                        "differentiates both. Add the value condition (the fields agree) to make a pair."
                    )
                assign[k] = diff[0]
        return assign

    def _fields(self, expr, derivative):
        """The (participant, field) pairs ``expr`` reads: every unknown, or only those it differentiates."""
        from ..trace import Hessian, Jacobian

        out = set()

        def key(n):
            for i, part in enumerate(self.parts):
                if part.owns(n):
                    return (i, getattr(n, "field_key", None) or id(getattr(n, "model", n)))
            return None

        def walk(n, inside):
            n = _unwrap(n)
            k = key(n)
            if k is not None and (inside or not derivative):
                out.add(k)
            for c in _iter(n):
                walk(c, inside or isinstance(n, (Jacobian, Hessian)))

        walk(expr, False)
        return out

    def _dirichlet_side(self, a, b, kv, kf):
        """The participant that receives the value condition of a Dirichlet-Neumann pair."""
        given = self.spec.dirichlet
        if given is not None:
            for i in (a, b):
                if given is self.parts[i].problem:
                    return i
            raise ValueError("jno.solve.coupling(dirichlet=...): pass one of the coupled problems.")
        takes = {i: getattr(self.parts[i], "takes_flux", True) for i in (a, b)}
        if not takes[a] and not takes[b]:
            raise NotImplementedError(
                "jno.core([...]): neither coupled side can receive a flux condition (two high-order jno.fem sides). "
                "Make one of them P1, or couple a high-order side to a jno.spectral / jno.fdm / jno.fvm / P1 one."
            )
        if takes[a] != takes[b]:  # the side that cannot take the flux takes the value
            return a if not takes[a] else b
        s = {i: self._stiffness(i, kv, kf) for i in (a, b)}
        return a if s[a] <= s[b] else b

    def _stiffness(self, i, kv, kf):
        """How much participant ``i``'s interface flux changes for a unit rise of its interface value, with the
        values of its OTHER interfaces held (the diagonal of its Steklov-Poincaré operator): solved with every
        value condition it takes part in as its Dirichlet condition, this one's data at 0 and at 1 (a linear
        probe; the mean over the interface, per unit weight). Held open, another interface would let a floating
        participant follow the probe and read no stiffness at all.

        A transient participant is probed over one window from its initial state, its data at what the senders
        send from theirs, this one's raised by a small step: the linearisation the first window sees. For a linear
        problem that is the response from rest; a nonlinear one (a gas, whose zero state has no density) has
        none at rest."""
        key = (i, kv)
        if key in self.stiffness:
            return self.stiffness[key]
        part = self.parts[i]
        cf = self.conditions[kf]
        mine = [k for k, c in enumerate(self.conditions) if not c.flux and i in c.owners and c.receiver is None]
        probe = self.__dict__.setdefault("_probe", {})
        if i not in probe:
            received = [(k, *self._receive(i, self.conditions[k])) for k in mine]
            for k, c, ds in received:
                for d in ds:
                    d.cond, d.received = k, c
            probe[i] = (part.rebuild([c for _, c, _ in received]), {k: d for k, _, d in received})
        prob, data = probe[i]
        iface = part.interface(self.conditions[kv].tags[i], self.spec.measure)
        flux = self._own_part(cf.expr, i)
        windowed = part.transient(prob) and self._window is not None
        about = windowed or i in self._guess  # probed about where it starts, not from zero
        starts = self._starts(windowed)
        mine = {lid: v for lid, v in self._probe_params.items() if lid in part.trainables(prob)} or None
        base = {}  # the data the probe starts from, per (condition, data index)
        for k, ds in data.items():
            face = part.interface(self.conditions[k].tags[i], self.spec.measure)
            for j, d in enumerate(ds):
                peers = [o for os in data.values() for o in os]
                base[k, j] = (
                    self._start_values(d, starts, peers, self._probe_params) if about else np.zeros(face.points.shape[0])
                )
        step = 1.0
        if about:  # a small step about the start for a nonlinear problem; exact either way for a linear one
            step = 1e-3 * max(1.0, max(float(np.abs(base[kv, j]).max(initial=0.0)) for j in range(len(data[kv]))))
        out = []
        for level in (0.0, step):
            extra = {}
            for k, ds in data.items():
                face = part.interface(self.conditions[k].tags[i], self.spec.measure)
                for j, d in enumerate(ds):
                    lv = base[k, j] + (level if k == kv else 0.0)
                    extra.update(d.setter(part.nodal(face, lv), prob))
            if windowed:
                window, t0, _ = self._window
                state, _ = part.march(prob, part.initial(prob), t0, window, extra, time=self._time_of(i), params=mine)
            else:
                state = part.solve(prob, extra, x0=self._guess.get(i), params=mine)
            out.append(np.asarray(part.sample(prob, state, flux, cf.tags[i], iface, extra, params=mine)))
        self.stiffness[key] = abs(float(iface.weights @ (out[1] - out[0]))) / (step * iface.total)
        return self.stiffness[key]

    def _initial_params(self):
        """``{layer_id: value}`` for the structural probes (the interface stiffness): a trainable parameter that still
        holds its placeholder, 0 until a jno.core setup applies its ``.initialize(...)``, is probed at its
        initializer's value instead. At 0 a trainable conductivity made the probed slab singular and put the value
        condition on the wrong side. Parameters set or trained already are probed where they are."""
        import jax
        import jax.numpy as jnp

        from ..trace import ModelCall

        found = {}
        for expr in [c for part in self.parts for c in part.constraints()] + [c.expr for c in self.conditions]:
            for n in _nodes(expr):
                model = getattr(n, "model", None) if isinstance(n, ModelCall) else None
                if model is None or not getattr(model, "_is_parameter", False) or model.layer_id in found:
                    continue
                init, value = getattr(model, "_initializer_fn", None), getattr(model.module, "value", None)
                if getattr(model, "_opt_fn", None) is None or init is None or value is None or np.any(np.asarray(value)):
                    continue
                key = getattr(model, "_initializer_key", None)
                key = jax.random.PRNGKey(0) if key is None else key
                try:
                    found[model.layer_id] = jnp.asarray(init(key, value.shape, value.dtype), value.dtype)
                except TypeError:  # an initializer without a dtype argument, as jno.core calls it too
                    found[model.layer_id] = jnp.asarray(init(key, value.shape), value.dtype)
        return found

    def _guesses(self):
        """``{participant: state}`` from ``coupling(x0={problem: state})``: where a steady participant starts (its first
        sweep and the stiffness probe). A gas has no state at zero, so a steady gas needs one."""
        given = self.spec.x0 or {}
        if not isinstance(given, dict):
            raise ValueError("jno.solve.coupling(x0=...): give a dict {problem: state}.")
        out = {}
        for prob, state in given.items():
            i = next((k for k, part in enumerate(self.parts) if part.problem is prob), None)
            if i is None:
                raise ValueError("jno.solve.coupling(x0=...): a problem that is not one of the coupled ones.")
            part = self.parts[i]
            if part.transient():
                raise ValueError(
                    "jno.solve.coupling(x0=...): a transient problem starts from its initial condition; x0= is where a "
                    "steady one starts."
                )
            if part.kind == "fem":
                raise NotImplementedError("jno.solve.coupling(x0=...): a jno.fem problem's steady solve takes no start.")
            v = np.asarray(state, dtype=float).reshape(-1)
            if v.size != int(part.problem._Ntot):
                raise ValueError(
                    f"jno.solve.coupling(x0=...): the state has {v.size} values, the problem {int(part.problem._Ntot)} "
                    "(one per unknown component and cell or node, as its solve(x0=) takes)."
                )
            out[i] = v
        return out

    def _starts(self, windowed=True):
        """Each participant's starting state: its initial state (transient, in a windowed coupling), its x0=, or None."""
        return [
            part.initial(part.problem) if windowed and part.transient() else self._guess.get(i)
            for i, part in enumerate(self.parts)
        ]

    def _start_values(self, d, starts, peers, params=None):
        """Where data field ``d`` starts, at its receiver's interface nodes. What its sender gives from the state it
        starts at (read on its own problem as built: the coupled ones may not exist yet); for a value whose sender has
        no state, the value that makes the received condition hold at the receiver's own start (the interface where
        the receiver already is: a gas starting at rest reads its wall at its own temperature, not at 0); else 0.
        ``peers`` are the data fields the receiver reads: a condition that also reads another one is left at 0."""
        import jax.numpy as jnp

        r = self.parts[d.receiver]
        dst = r.interface(d.receiver_tag, self.spec.measure)
        s = self.parts[d.sender]
        if starts[d.sender] is not None:
            src = s.interface(d.sender_tag, self.spec.measure)
            mine = self._params_of(d.sender, params)
            vals = s.sample(s.problem, starts[d.sender], d.expr, d.sender_tag, src, params=mine)
            return self._map_jnp(d, dst, src) @ jnp.asarray(vals, float)
        zero = np.zeros(dst.points.shape[0])
        if d.flux or starts[d.receiver] is None or d.received is None:
            return zero
        coef, rest = self._solve_for(d)
        if coef is None:
            return zero
        if rest is not None and any(_unwrap(n) is _unwrap(o.node) for o in peers for n in _nodes(rest)):
            return zero
        if rest is None:
            return zero
        mine = self._params_of(d.receiver, params)
        vals = r.sample(r.problem, starts[d.receiver], rest, d.receiver_tag, dst, params=mine)
        return jnp.broadcast_to(-jnp.asarray(vals, float) / coef, zero.shape)

    def _solve_for(self, d):
        """``(c, rest)`` with the received condition ``= c * d.node + rest`` and ``rest`` free of ``d.node``; ``(None,
        None)`` when it is not affine in it in that plain way (the node inside a product of unknowns, a function)."""
        from .._fvm.trace import _concrete_literal, _factors, _signed_terms

        coef, rest = 0.0, []
        for sg, term in _signed_terms(d.received, distribute=False):
            factors = _factors(term)
            hits = [f for f in factors if _unwrap(f) is _unwrap(d.node)]
            if not hits:
                if any(_unwrap(n) is _unwrap(d.node) for n in _nodes(term)):
                    return None, None
                rest.append(term if sg > 0 else -1.0 * term)
                continue
            others = [f for f in factors if f is not hits[0]]
            if len(hits) > 1 or not all(_concrete_literal(f) for f in others):
                return None, None
            coef += sg * float(np.prod([float(np.asarray(_unwrap(f).value)) for f in others]))
        if coef == 0.0:
            return None, None
        total = None
        for t in rest:
            total = t if total is None else total + t
        return coef, total

    def _start_data(self, states, params=None):
        """Every data field's values at its start (:meth:`_start_values`) before the first sweep: one array per field,
        in the order of ``self.data``, at its receiver's interface nodes."""
        return tuple(
            self._start_values(d, states, [o for o in self.data if o.receiver == d.receiver], params) for d in self.data
        )

    def _params_of(self, i, params):
        """The trainable parameters among ``params`` (``{layer_id: value}``) that participant ``i`` reads."""
        if not params:
            return None
        problems = self.__dict__.get("problems")  # built after the structural probes, which read the originals
        own = self.parts[i].trainables(problems[i] if problems is not None else self.parts[i].problem)
        mine = {lid: v for lid, v in params.items() if lid in own}
        return mine or None

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
            for d in data:
                d.cond, d.received = k, cond
            self._all_data.extend(data)
            extra[r].append(self._as_condition(r, cond, c))
        problems = [p.rebuild(extra[i]) if extra[i] else p.problem for i, p in enumerate(self.parts)]
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
        test = self._test_function(i, original)
        field = part._flux_field(self._own_part(original.expr, i))
        shape = tuple(getattr(field, "value_shape", ()) or ())
        if shape:  # a traction: the load against the vector test function, component by component
            term = _componentwise(rest, test, shape[0])
        else:
            term = rest * test
        from .transfer import radial_axis

        axis = radial_axis(self.spec.measure)
        if axis is not None:  # the forms carry 2 pi r ds
            r = [v for k, v in original.coords[i].items() if int(getattr(v, "dim", [0])[0]) == axis]
            if len(r) != 1:
                raise ValueError("jno.solve.coupling(measure='axisymmetric'): the interface binding has no r coordinate.")
            term = term * (2.0 * np.pi * r[0])
        elif self.spec.measure is not None:
            raise NotImplementedError(
                "jno.solve.coupling(measure=<callable>): a finite-element side writes its interface load with the "
                "measure as an expression; use None or 'axisymmetric'."
            )
        return term

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
    def _produce(self, d, states, values, frames=None, params=None):
        """Data ``d`` from its sender's current state, carried to the receiver's interface nodes; ``values`` the data
        the sender read (its boundary conditions, which its own samples read too), ``frames`` the window's frames of
        the participants that marched one (:meth:`Participant.march`). Traceable.

        A flux from a sender that just marched a window is its mean over the window, from the sender's own
        steps: what the receiver takes in over the window is then what the sender gave out. Its end value is
        not, when the sender takes several steps (an explicit march, a finer step) while the flux changes."""
        import jax.numpy as jnp

        s, r = self.parts[d.sender], self.parts[d.receiver]
        src = s.interface(d.sender_tag, self.spec.measure)
        dst = r.interface(d.receiver_tag, self.spec.measure)
        prob, data = self.problems[d.sender], self._extra(d.sender, values)
        mine = self._params_of(d.sender, params)
        fr = (frames or {}).get(d.sender) if d.flux else None
        if fr is None or fr.shape[0] < 2:
            vals = jnp.asarray(s.sample(prob, states[d.sender], d.expr, d.sender_tag, src, data, params=mine))
        else:
            w = self._window_weights(d.sender, fr.shape[0] - 1)
            vals = sum(
                wk * jnp.asarray(s.sample(prob, fr[k], d.expr, d.sender_tag, src, data, params=mine))
                for k, wk in enumerate(w)
                if wk != 0.0
            )
        return self._map_jnp(d, dst, src) @ vals

    def _window_weights(self, i, m):
        """Weights of a window's ``m + 1`` frames in its mean: the trapezoid rule for an explicit (or IMEX) march,
        whose flux is continuous across its steps, the step ends for an implicit Euler step, whose step takes the
        flux at its end -- exactly what it gave out."""
        from ..utils.solver.imex import IMEXScheme
        from ..utils.solver.timeschemes import _ERKScheme

        w = np.full(m + 1, 1.0 / m)
        if isinstance(self._time_of(i), (_ERKScheme, IMEXScheme)) or getattr(self._time_of(i), "continuous_flux", False):
            w[0] = w[-1] = 0.5 / m
        else:
            w[0] = 0.0
        return w

    def _map(self, d, dst, src):
        cache = self.__dict__.setdefault("_maps", {})
        key = (d.receiver, d.receiver_tag, d.sender, d.sender_tag, d.flux)
        if key not in cache:
            send, recv = self.parts[d.sender], self.parts[d.receiver]
            interp = getattr(send, "spectral_interp", None) or getattr(send, "edge_interp", None)
            if interp is not None and (not d.flux or getattr(recv, "kind", None) == "spectral"):
                # a spectral sender's interpolant read at the receiver's nodes: spectrally accurate (a flux too, when
                # the receiver is spectral: both sides hold it pointwise)
                import scipy.sparse as _sp

                cache[key] = _sp.csr_matrix(interp(d.sender_tag, dst.points))
            else:
                cache[key] = (conservative if d.flux else consistent)(dst, src)
        return cache[key]

    def _map_jnp(self, d, dst, src):
        """:meth:`_map` as a sparse JAX matrix (BCOO), for the traceable transfer."""
        from .transfer import _Projection, to_jax

        cache = self.__dict__.setdefault("_maps_jnp", {})
        key = (d.receiver, d.receiver_tag, d.sender, d.sender_tag, d.flux)
        if key not in cache:
            M = self._map(d, dst, src)
            cache[key] = M if isinstance(M, _Projection) else to_jax(M)  # a projection is traceable itself
        return cache[key]

    def _extra(self, i, values):
        """What participant ``i``'s solve takes for the data fields it reads, from their ``values``."""
        part, prob = self.parts[i], self.problems[i]
        extra = {}
        for d, v in zip(self.data, values):
            if d.receiver == i:
                iface = part.interface(d.receiver_tag, self.spec.measure)
                extra.update(d.setter(part.nodal(iface, v), prob))
        return extra

    def _advance(self, i, start, states, values, t, window, params=None):
        """Participant ``i`` with the data ``values``: a window's march from ``start`` for a transient problem, a
        (warm-started) solve otherwise -- a steady problem in a transient coupling is quasi-steady. Returns its
        state and the window's frames (``None`` for a solve)."""
        part, prob = self.parts[i], self.problems[i]
        extra, mine = self._extra(i, values), self._params_of(i, params)
        if window is not None and part.transient(prob):
            return part.march(prob, start[i], t, window, extra, time=self._time_of(i), params=mine)
        return part.solve(prob, extra, x0=states[i], params=mine), None

    def _time_of(self, i):
        tm = self.spec.time
        if isinstance(tm, dict):
            return next((s for p, s in tm.items() if p is self.parts[i].problem), None)
        return tm

    def _back(self):
        """Indices of the data fields read before their sender runs in a sweep (every one, for Jacobi): the
        iteration's unknowns."""
        if self.spec.scheme == "jacobi":
            return list(range(len(self.data)))
        pos = {i: k for k, i in enumerate(self.order)}
        return [k for k, d in enumerate(self.data) if pos[d.sender] >= pos[d.receiver]]

    def _sweep(self, x, start, states, values, back, t=None, window=None, params=None):
        """One sweep from the back data ``x``: every participant advanced once. Returns the states, the data values
        each participant read, and the back data its senders produce from them."""
        import jax.numpy as jnp

        values = self._unpack(values, back, x)
        n = len(self.parts)
        states, frames = list(states), {}
        if self.spec.scheme == "jacobi":
            advanced = [self._advance(i, start, states, values, t, window, params) for i in range(n)]
            for i, (state, fr) in enumerate(advanced):
                states[i], frames[i] = state, fr
        else:
            for i in self.order:
                fresh = {  # forward data: fresh from senders already run in this sweep
                    k: self._produce(d, states, values, frames, params)
                    for k, d in enumerate(self.data)
                    if d.receiver == i and k not in back
                }
                values = tuple(fresh.get(k, v) for k, v in enumerate(values))
                states[i], frames[i] = self._advance(i, start, states, values, t, window, params)
        produced = [self._produce(self.data[k], states, values, frames, params) for k in back]
        return states, values, jnp.concatenate(produced) if back else jnp.zeros(0)

    def _iterate(self, start, states, values, t=None, window=None, params=None):
        """One coupled solve -- the steady problem, or one window -- iterated until the interface data stop
        changing (or a single sweep, ``implicit=False``). Returns the participants' states and the data values."""
        states = list(states)
        back = self._back()
        x = np.concatenate([np.asarray(values[k]) for k in back]) if back else np.zeros(0)
        self._accel.start()
        history = []
        for it in range(1, self.spec.max_iter + 1):
            states, values, x_new = self._sweep(x, start, states, values, back, t, window, params)
            x_new = np.asarray(x_new)
            err = float(np.linalg.norm(x_new - x)) / max(float(np.linalg.norm(x_new)), 1e-300)
            history.append(err)
            self.history = history
            if err <= self.spec.tol or not back or not self.spec.implicit:
                values = self._unpack(values, back, x_new)
                break
            if _stalled(history):
                raise self._stall_error(err)
            x = self._accel.next(x, x_new)
        else:
            raise self._unconverged_error(t)
        return states, values

    def _stall_error(self, err):
        return RuntimeError(
            f"jno.core([...]): the coupling stalled at an interface change of {err:.1e} (history "
            f"{self._brief()}), above tol = {self.spec.tol:.1e}: the participants' own solves are not that "
            "accurate. Set tol above it, or solve the participants more accurately (a direct solver)."
        )

    def _unconverged_error(self, t):
        return RuntimeError(
            f"jno.core([...]): the coupling did not converge in {self.spec.max_iter} iterations"
            f"{'' if t is None else f' in the window at t = {t:g}'} (interface change {self.history[-1]:.2e} > tol "
            f"{self.spec.tol:.1e}; history {self._brief()}). Raise max_iter, accelerate (accel='iqn-ils' or "
            "'aitken'), or swap the Dirichlet side (dirichlet=...)."
        )

    def _verdict(self, aux, t=None):
        """The step-by-step iteration's verdict on a traced fixed point, from what it returned: ``(iterations, last
        change, history)``. Concrete, it raises here; traced, from a callback when the program runs."""
        import jax

        def judge(it, err, hist):
            self.history = [float(e) for e in np.asarray(hist)[: int(it)]]
            if self._one_sweep() or not self.history or self.history[-1] <= self.spec.tol:
                return  # one sweep per window by design, or converged
            if _stalled(self.history):
                raise self._stall_error(self.history[-1])
            raise self._unconverged_error(t)

        if _tracing(*aux):
            jax.debug.callback(judge, *aux)
        else:
            judge(*aux)

    def _fixed_point(self, start, states, values, back, t, window, params, differentiable=True):
        """The interface fixed point of one window (or of the steady problem) as one traceable computation: the sweeps
        in a ``lax.while_loop`` with Aitken's relaxation (or a fixed one) carried in jnp, until the change of the back
        data is below ``tol`` (or it stalls, or ``max_iter``). Its derivative is the implicit one at the root
        (``custom_root``): the unknown is the back data with every participant's state, ``z = (x, s)``, the root of
        ``z - (G(x), S(x))``, and a tangent is one GMRES solve with that map's Jacobian (in a ``custom_linear_solve``,
        so reverse mode solves the transposed system). ``differentiable=False`` (a compiled window called with concrete
        inputs, which nothing differentiates) runs the loop alone. Returns the states, the data values and
        ``(iterations, last change, history)`` for :meth:`_verdict`."""
        import jax
        import jax.numpy as jnp
        from jax.flatten_util import ravel_pytree
        from jax.scipy.sparse.linalg import gmres

        if not isinstance(self._accel, _Aitken):
            raise NotImplementedError(
                "jno.solve.coupling(accel='iqn-ils'): a compiled or differentiated coupling iterates with accel='aitken' "
                "or a fixed relaxation; IQN-ILS runs step by step only (git-bug 1c39855)."
            )
        omega0, fixed = float(self._accel.omega0), self._accel.fixed
        tol, max_iter = float(self.spec.tol), int(self.spec.max_iter)
        x0 = jnp.concatenate([jnp.asarray(values[k], float) for k in back]) if back else jnp.zeros(0)

        def sweep(x, warm):
            return self._sweep(x, start, warm, values, back, t, window, params)

        if any(w is None for w in states):  # no warm start: zeros shaped like what the participant returns
            shapes = jax.eval_shape(lambda x: sweep(x, states)[0], x0)
            states = [jnp.zeros(sh.shape, sh.dtype) if w is None else w for w, sh in zip(states, shapes)]
        states = [jnp.asarray(w) for w in states]

        def relax(x, x_new, r_prev, omega, first):
            r = x_new - x
            if fixed is not None:
                return x + fixed * r, r, omega
            dr = r - r_prev
            denom = dr @ dr
            new = jnp.where(denom > 0.0, -omega * (r_prev @ dr) / jnp.where(denom > 0.0, denom, 1.0), omega)
            omega = jnp.where(first, omega0, new)
            return x + omega * r, r, omega

        def forward(_f, z0):
            def body(c):
                it, x, warm, r_prev, omega, err, hist = c
                warm, _, x_new = sweep(x, warm)
                err = jnp.linalg.norm(x_new - x) / jnp.maximum(jnp.linalg.norm(x_new), 1e-300)
                hist = hist.at[it].set(err)
                x_relaxed, r_prev, omega = relax(x, x_new, r_prev, omega, it == 0)
                return it + 1, jnp.where(err <= tol, x_new, x_relaxed), warm, r_prev, omega, err, hist

            def cond(c):
                it, err, hist = c[0], c[5], c[6]
                recent = jax.lax.dynamic_slice(hist, (jnp.maximum(it - 3, 0),), (3,))
                stalled = (it >= 4) & (jnp.max(recent) <= 1.5 * jnp.min(recent)) & (err < 1e-4)
                return (it < max_iter) & (err > tol) & ~stalled

            x, warm = z0
            init = (
                jnp.int32(0),
                x,
                warm,
                jnp.zeros_like(x),
                jnp.asarray(omega0),
                jnp.asarray(jnp.inf),
                jnp.zeros(max(max_iter, 3)),
            )
            it, x, warm, _, _, err, hist = jax.lax.while_loop(cond, body, init)
            return (x, warm), (it.astype(err.dtype), err, hist)  # all floating: custom_root's jvp rejects an int aux

        def residual(z):
            x, warm = z
            new, _, x_new = sweep(x, [jax.lax.stop_gradient(w) for w in warm])  # a warm start changes no root
            return x - x_new, [w - wn for w, wn in zip(warm, new)]

        def tangent(g, y):
            # on one flat vector: custom_linear_solve's transpose wants every right-hand-side leaf linear in the input,
            # and a block whose tangent is a constant zero (a state no parameter reaches) is not
            flat, unravel = ravel_pytree(y)

            def krylov(mv, b):
                return gmres(mv, b, tol=1e-12, atol=0.0, restart=30, maxiter=20)[0]

            matvec = lambda v: ravel_pytree(g(unravel(v)))[0]  # noqa: E731
            return unravel(jax.lax.custom_linear_solve(matvec, flat, krylov, transpose_solve=krylov))

        if differentiable:
            (x, warm), aux = jax.lax.custom_root(residual, (x0, states), forward, tangent, has_aux=True)
        else:
            (x, warm), aux = forward(None, (x0, states))
        return warm, self._unpack(values, back, x), aux

    def _windows(self, transient):
        """``(window, t0, t1)`` of a transient coupling: the transient problems' common time span, the window the
        given one or the largest of their steps, a whole number of every one of them."""
        from ..utils.solver.time_route import _infer_time_window

        spans = [_infer_time_window(self.parts[i].domain) for i in transient]
        t0, t1 = spans[0][0], spans[0][1]
        if any(abs(a - t0) > 1e-12 or abs(b - t1) > 1e-12 for a, b, _ in spans):
            raise ValueError(
                "jno.core([...]): the coupled transient problems march different time spans "
                f"({[(a, b) for a, b, _ in spans]}); give their domains the same time=(t0, t1, ...)."
            )
        window = self.spec.window if self.spec.window is not None else max(dt for _, _, dt in spans)
        for _, _, dt in spans:
            k = window / dt
            if abs(k - round(k)) > 1e-9 * max(1.0, k):
                raise ValueError(
                    f"jno.core([...]): the coupling window {window:g} is not a whole number of a problem's step "
                    f"{dt:g}; set jno.solve.coupling(window=...) to a common multiple."
                )
        n = (t1 - t0) / window
        if abs(n - round(n)) > 1e-9 * max(1.0, n):
            raise ValueError(f"jno.core([...]): the time span {t1 - t0:g} is not a whole number of windows {window:g}.")
        return float(window), float(t0), float(t1)

    def solve(self, params=None):
        """Iterate to the coupled solution: a list with each problem's solution, in the order given -- for a
        transient coupling, each problem's states at the window ends (the initial state first).

        ``params`` (``{layer_id: value}``) gives the participants' trainable parameters their values. Traced (under
        ``jax.grad``, ``jax.jit``), every window's fixed point is one traced computation, differentiated implicitly
        at its root (:meth:`_fixed_point`), and the windows chain through their states."""
        import jax
        import jax.numpy as jnp

        n = len(self.parts)
        self._accel = _accelerator(self.spec)
        if params:
            known = set().union(*(p.trainables(q) for p, q in zip(self.parts, self.problems)))
            unknown = sorted(set(params) - known)
            if unknown:
                raise ValueError(f"coupling: params= names layer ids {unknown}, which no coupled problem trains.")
        traced = _tracing(params)
        transient = [i for i in range(n) if self.parts[i].transient(self.problems[i])]
        stack = jnp.stack if traced else np.stack
        if not transient:
            states = [self._guess.get(i) for i in range(n)]
            values = self._start_data(states, params)
            if traced:
                states, _, aux = self._fixed_point([None] * n, states, values, self._back(), None, None, params)
                self._verdict(aux)
            else:
                states, _ = self._iterate([None] * n, states, values, params=params)
                self.iterations = len(self.history)
                self.states = states
            return [self._shape(i, st) for i, st in enumerate(states)]
        window, t0, t1 = self._window
        start = [self.parts[i].initial(self.problems[i]) if i in transient else None for i in range(n)]
        states = [start[i] if i in transient else self._guess.get(i) for i in range(n)]
        values = self._start_data(states, params)
        frames = [[] for _ in range(n)]
        self.window_iterations, self.times = [], [t0]
        run = self._window_run(transient)
        program = None
        if not traced and self._compilable(transient):  # concrete: nothing differentiates it, the loop alone
            eager_run = self._window_run(transient, differentiable=False)
            program = jax.jit(lambda s, v, p: eager_run(s, v, p, t0))  # autonomous: every window from t0's program
        t = t0
        for w in range(int(round((t1 - t0) / window))):
            if traced:  # every window traced, its fixed point differentiated at its root
                states, values, aux = run(states, values, params, t)
                self._verdict(aux, t)
            elif program is not None and w > 0:  # the first window step by step: it makes the structural decisions
                states, values, aux = program(states, values, params)
                self._verdict(aux, t)
            else:
                states, values = self._iterate(start, states, values, t, window, params)
            if w == 0:  # a quasi-steady problem's first frame is its state in the first window
                for i in range(n):
                    frames[i].append(start[i] if i in transient else states[i])
            for i in range(n):
                frames[i].append(states[i])
            if not traced:
                self.window_iterations.append(len(self.history))
            start = [states[i] if i in transient else None for i in range(n)]
            t += window
            self.times.append(t)
        if not traced:
            self.iterations = int(sum(self.window_iterations))
            self.states = states
        return [stack([self._shape(i, f) for f in frames[i]]) for i in range(n)]

    def trainable_nodes(self):
        """``[(layer_id, node)]`` of the trainable parameters the coupled problems read, each once (a parameter in an
        interface condition is read by the problem that receives it too)."""
        found = {}
        for part, prob in zip(self.parts, self.problems):
            for lid, node in part.param_nodes(prob):
                found.setdefault(lid, node)
        return list(found.items())

    def nodes(self):
        """The coupled solution as trace nodes of its trainable parameters, one per problem in the order given: a
        ``jno.core`` loss on them trains the parameters through the coupling. They share one node that runs the
        coupled solve (:meth:`solve` with ``params=``) once per evaluation."""
        import jax
        import jax.numpy as jnp

        from ..trace import FunctionCall, ModelWeights

        specs = self.trainable_nodes()
        lids = [lid for lid, _ in specs]

        def solved(*values):
            return self.solve(params=dict(zip(lids, values)))

        def stored(node):  # what the node evaluates to, for the shapes only
            n = _unwrap(node)
            return n.model.module if isinstance(n, ModelWeights) else jnp.asarray(n.model.module.value)

        shapes = jax.eval_shape(solved, *[stored(n) for _, n in specs])
        whole = FunctionCall(
            lambda *v: jnp.concatenate([jnp.ravel(jnp.asarray(o)) for o in solved(*v)]),
            [n for _, n in specs],
            name="coupled_solve",
        )
        whole._per_evaluation = True  # one coupled solve per evaluation, whichever problems the loss reads
        whole._domain = self.parts[0].domain
        out, at = [], 0
        for i, sh in enumerate(shapes):
            size = int(np.prod(sh.shape))
            node = FunctionCall(
                lambda w, at=at, size=size, shape=tuple(sh.shape): w[at : at + size].reshape(shape),
                [whole],
                name=f"coupled_solve[{i}]",
            )
            node._domain = self.parts[i].domain  # so jno.core infers the domain of a loss on this problem
            out.append(node)
            at += size
        return out

    def _one_sweep(self):
        """Is every window one sweep: data flowing one way (no back data), or a loose coupling?"""
        return not self._back() or not self.spec.implicit

    def _window_run(self, transient, differentiable=True):
        """One window as one traceable function ``(states, values, params, t) -> (states, values, aux)``: one sweep
        (:meth:`_one_sweep`), or the interface fixed point (:meth:`_fixed_point`). ``self.compiles`` counts its
        traces."""
        import jax.numpy as jnp

        window = self._window[0]
        back = self._back()
        one = self._one_sweep()

        def run(states, values, params, t):
            self.compiles += 1
            start = [states[i] if i in transient else None for i in range(len(self.parts))]
            if not one:
                return self._fixed_point(start, states, values, back, t, window, params, differentiable)
            x = jnp.concatenate([jnp.asarray(values[k]) for k in back]) if back else jnp.zeros(0)
            states, values, x_new = self._sweep(x, start, states, values, back, t, window, params)
            err = jnp.linalg.norm(x_new - x) / jnp.maximum(jnp.linalg.norm(x_new), 1e-300)
            return states, self._unpack(values, back, x_new), (jnp.ones((), err.dtype), err, jnp.reshape(err, (1,)))

        return run

    def _compilable(self, transient):
        """Can every window after the first run as one compiled program: asked for (``compile=``), windows that do
        not depend on their start time, and an iteration that runs in jnp (one sweep, or Aitken)?"""
        if not self.spec.compile or not all(self.parts[i].autonomous() for i in transient):
            return False
        return self._one_sweep() or isinstance(self._accel, _Aitken)

    def _brief(self):
        h = self.history
        return ", ".join(f"{e:.1e}" for e in (h[:3] + ["…"] + h[-2:] if len(h) > 6 else h) if not isinstance(e, str))

    @staticmethod
    def _unpack(values, back, x):
        """``values`` with the back data fields (indices ``back``) replaced by their slices of ``x``."""
        out, k = list(values), 0
        for j in back:
            m = np.size(values[j])
            out[j] = x[k : k + m]
            k += m
        return tuple(out)

    def _shape(self, i, state):
        import jax

        p = self.problems[i]
        part = self.parts[i]
        state = state if isinstance(state, jax.core.Tracer) else np.asarray(state)
        if part.kind == "fem":
            return state
        nf, N = getattr(p, "_nf", 1), getattr(p, "_N", None)
        return state.reshape(N) if nf == 1 else state.reshape(nf, N)


def _stalled(history):
    """Have the last three changes stopped falling (within a factor 1.5) below 1e-4: the participants' own accuracy?"""
    return len(history) >= 4 and max(history[-3:]) <= 1.5 * min(history[-3:]) and history[-1] < 1e-4


def _componentwise(rest, test, dim):
    """``rest · test`` for a vector load ``rest`` (sums of scalar coefficients times a boundary normal), written
    ``Σ_k coef · n[..., k] · test[k]``. Written as ``inner(coef * n, test)`` it is wrong in jno.fem when the
    coefficient varies along the facet (git-bug 8388acd)."""
    from .._fvm.trace import _factors, _is_normal, _product, _signed_terms
    from ..trace import Variable

    total = None
    for sign, term in _signed_terms(rest, distribute=True):
        fs = _factors(term)
        normals = [f for f in fs if isinstance(_unwrap(f), Variable) and _is_normal(_unwrap(f))]
        if len(normals) != 1:
            raise NotImplementedError(
                "jno.core([...]): a vector load on a finite-element side is read as a coefficient times the boundary "
                "normal (p * n); this one is another vector expression."
            )
        coef = _product([f for f in fs if f is not normals[0]])
        for k in range(dim):
            piece = coef * normals[0][..., k] * test[k]
            piece = piece if sign > 0 else -1.0 * piece
            total = piece if total is None else total + piece
    return total


class _Aitken:
    """Dynamic relaxation ``x + ω r``, ``ω`` from the last two residuals (Küttler & Wall 2008); a fixed ``ω`` when
    one is given."""

    def __init__(self, omega0, fixed=None):
        self.omega0, self.fixed, self.omega, self.r_prev = omega0, fixed, None, None

    def start(self):
        self.r_prev = None

    def next(self, x, x_new):
        r = x_new - x
        if self.fixed is not None:
            return x + self.fixed * r
        if self.r_prev is None or self.omega is None:
            self.omega = self.omega0
        else:
            dr = r - self.r_prev
            denom = float(dr @ dr)
            if denom > 0.0:
                self.omega = float(-self.omega * (self.r_prev @ dr) / denom)
        self.r_prev = r
        return x + self.omega * r


class _IQNILS:
    """Interface quasi-Newton with an inverse Jacobian from a least-squares model (IQN-ILS, Degroote, Bathe &
    Vierendeels 2009): ``x = x̃ + W c``, ``c = argmin ‖V c + r‖``, ``V`` and ``W`` the differences of the residuals
    and of the outputs this window (and of the last ``reuse`` windows'). Columns that would make the
    least-squares problem singular are dropped (QR filter)."""

    def __init__(self, omega0, reuse=0, eps=1e-10):
        self.omega0, self.reuse, self.eps = omega0, int(reuse), eps
        self.V, self.W, self.old, self.prev = [], [], [], None

    def start(self):
        if self.V:
            self.old.append((self.V, self.W))
            self.old = self.old[-self.reuse :] if self.reuse else []
        self.V, self.W, self.prev = [], [], None

    def next(self, x, x_new):
        r = x_new - x
        if self.prev is not None:
            self.V.insert(0, r - self.prev[1])
            self.W.insert(0, x_new - self.prev[0])
        self.prev = (x_new, r)
        V = self.V + [v for vs, _ in self.old for v in vs]
        W = self.W + [w for _, ws in self.old for w in ws]
        if not V:
            return x + self.omega0 * r
        Vm, Wm = np.stack(V, axis=1), np.stack(W, axis=1)
        while Vm.shape[1]:
            Q, R = np.linalg.qr(Vm)
            d = np.abs(np.diag(R))
            bad = np.flatnonzero(d < self.eps * max(float(np.linalg.norm(R)), 1e-300))
            if bad.size == 0:
                break
            Vm, Wm = np.delete(Vm, bad[0], axis=1), np.delete(Wm, bad[0], axis=1)
        if not Vm.shape[1]:
            return x + self.omega0 * r
        c = np.linalg.solve(R, -(Q.T @ r))
        return x_new + Wm @ c


def _accelerator(spec):
    a = spec.accel
    if a is None:
        return _Aitken(1.0, fixed=1.0)
    if isinstance(a, (int, float)):
        return _Aitken(float(a), fixed=float(a))
    if a == "iqn-ils":
        return _IQNILS(spec.omega0, reuse=spec.reuse)
    return _Aitken(spec.omega0)


def detect(constraints):
    """``(problems, conditions)`` when a ``jno.core`` list holds two or more problems on their own meshes (no
    shared-mesh region; those are ``jno.dd``'s) and, besides them, only conditions; else ``None``."""
    from .._fem import FEM
    from .._fvm.trace import _TraceFVM
    from .._spectral.trace import _TraceSpectral
    from ..fdm import _TraceFDM

    kinds = (FEM, _TraceFVM, _TraceFDM, _TraceSpectral)
    problems = [c for c in constraints if isinstance(c, kinds)]
    if len(problems) < 2 or any(getattr(p, "region_geometry", None) is not None for p in problems):
        return None
    return problems, [c for c in constraints if not isinstance(c, kinds)]


def couple(problems, conditions, spec=None):
    """The coupled problem of ``problems`` (each on its own mesh) joined by the interface ``conditions``."""
    return Coupled(problems, conditions, spec)
