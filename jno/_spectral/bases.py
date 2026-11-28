"""One-dimensional spectral bases: nodes, quadrature, differentiation, interpolation and transforms.

Based on:
  Trefethen, L. N. (2000). Spectral Methods in MATLAB. SIAM -- Fourier and Chebyshev differentiation
  (ch. 3, 6), Clenshaw-Curtis weights (ch. 12).
  Berrut, J.-P. & Trefethen, L. N. (2004). Barycentric Lagrange interpolation. SIAM Review 46(3), 501-517 --
  the interpolation matrix and the differentiation matrix of any node set from one set of weights.
  Baltensperger, R. & Trummer, M. R. (2003). Spectral differencing with a twist. SIAM J. Sci. Comput. 24(5),
  1465-1487 -- the diagonal as minus the off-diagonal row sum (the "negative sum trick").
  Canuto, C., Hussaini, M. Y., Quarteroni, A. & Zang, T. A. (2006). Spectral Methods: Fundamentals in Single
  Domains. Springer -- Gauss and Gauss-Lobatto rules, the discrete Chebyshev transform, the derivative
  recurrence for Chebyshev coefficients (sec. 2.4).
  Golub, G. H. & Welsch, J. H. (1969). Calculation of Gauss quadrature rules. Math. Comp. 23, 221-230.

Everything that depends only on the node set (nodes, weights, matrices) is built once on the host in float64
and cached on the instance; applying it is ``jax`` and differentiable. Operators act along one axis of an
array, so a tensor-product field is never multiplied by a Kronecker matrix (see :mod:`jno._spectral.grid`).

Conventions:
  * nodes ascend, so the first node of an axis is its "left"/"bottom"/"front" face;
  * a Fourier axis has ``n`` nodes on ``[a, b)``: the right end is the periodic image of the left, not a node;
  * a Chebyshev or Legendre axis of size ``n`` holds polynomials of degree ``n - 1``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["Axis", "Fourier", "Chebyshev", "Legendre", "apply_axis", "axis_from_spec"]


# ── helpers ─────────────────────────────────────────────────────────────────────────────────────────────


def _inexact(u):
    u = jnp.asarray(u)
    return u if jnp.issubdtype(u.dtype, jnp.inexact) else u.astype(jnp.result_type(float))


#: Every matrix product of the spectral package asks for full float32 precision. On an Ampere-or-later GPU a
#: float32 ``dot_general`` otherwise runs in TensorFloat-32 (a 10-bit mantissa): a differentiation matrix applied
#: that way loses the accuracy the basis exists for (measured on the sum-factorised elements of ``jno.fem``, RTX
#: 3070: an operator apply off by 5.6e-2 at 2M DOFs against 2e-6). float64 is unaffected.
HIGHEST = jax.lax.Precision.HIGHEST


def _real_dtype(u):
    return jnp.finfo(u.dtype).dtype


def apply_axis(M, u, axis: int):
    """``M`` (``m x n``, host or device) applied along ``axis`` of ``u``: ``u`` contracted with ``M[:, :]``."""
    u = _inexact(u)
    M = jnp.asarray(M, dtype=_real_dtype(u)) if isinstance(M, np.ndarray) and not np.iscomplexobj(M) else M
    out = jnp.tensordot(M, jnp.moveaxis(u, axis, 0), axes=(1, 0), precision=HIGHEST)
    return jnp.moveaxis(out, 0, axis)


def _shape_along(n, axis, ndim):
    return [n if i == axis else 1 for i in range(ndim)]


class Axis:
    """A one-dimensional spectral basis on ``[a, b]``. Subclasses fill the node set and the transforms."""

    family: str = ""
    periodic: bool = False

    def __init__(self, n: int, a: float = -1.0, b: float = 1.0):
        n = int(n)
        if n < self._min_n:
            raise ValueError(f"jno.spectral: a {self.family} axis needs at least {self._min_n} nodes, got n={n}.")
        a, b = float(a), float(b)
        if not b > a:
            raise ValueError(f"jno.spectral: the interval of a {self.family} axis must have b > a, got [{a}, {b}].")
        self.n, self.a, self.b = n, a, b

    _min_n = 1

    # identity: two axes with the same key are the same operator set (used to share caches and skip transfers)
    @property
    def key(self) -> tuple:
        return (self.family, self.rule, self.n, self.a, self.b)

    rule: str = ""

    def __eq__(self, other):
        return isinstance(other, Axis) and self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __repr__(self):
        rule = f", rule={self.rule!r}" if self.rule else ""
        return f"{type(self).__name__}(n={self.n}, a={self.a:g}, b={self.b:g}{rule})"

    @property
    def length(self) -> float:
        return self.b - self.a

    def with_n(self, n: int) -> "Axis":
        """The same family, rule and interval with ``n`` nodes."""
        return type(self)(n, self.a, self.b, **self._extra())

    def _extra(self) -> dict:
        return {}

    # host-side data, overridden
    @functools.cached_property
    def nodes(self) -> np.ndarray:
        raise NotImplementedError

    @functools.cached_property
    def weights(self) -> np.ndarray:
        """Quadrature weights: ``∫_a^b f dx ≈ Σ w_j f(x_j)``."""
        raise NotImplementedError

    def interp_matrix(self, x) -> np.ndarray:
        """``(len(x), n)``: the values at ``x`` of the interpolant through the nodes."""
        raise NotImplementedError

    def diff_matrix(self, order: int = 1) -> np.ndarray:
        """``(n, n)``: the ``order``-th derivative of the interpolant, at the nodes."""
        return self._diff_matrix(int(order))

    def _diff_matrix(self, order):
        eye = np.eye(self.n)
        return np.asarray(self.diff(eye, order, axis=0))

    def integrate(self, u, axis: int = 0):
        """``∫_a^b u dx`` along ``axis`` by the node quadrature."""
        u = _inexact(u)
        w = jnp.asarray(self.weights, dtype=_real_dtype(u))
        return jnp.tensordot(w, jnp.moveaxis(u, axis, 0), axes=(0, 0), precision=HIGHEST)

    def interp(self, u, x, axis: int = 0):
        """The interpolant of ``u`` (nodal values along ``axis``) evaluated at the points ``x``."""
        return apply_axis(self.interp_matrix(np.asarray(x, dtype=float).reshape(-1)), u, axis)

    def to(self, u, other: "Axis", axis: int = 0):
        """Nodal values on ``other`` (same interval) of the interpolant of ``u``. Identity when ``other == self``."""
        if other == self:
            return _inexact(u)
        if (other.a, other.b) != (self.a, self.b):
            raise ValueError(f"jno.spectral: cannot move a field from {self} to {other}: the intervals differ.")
        return self.interp(u, other.nodes, axis)


# ── Fourier ─────────────────────────────────────────────────────────────────────────────────────────────


class Fourier(Axis):
    """Trigonometric interpolation on ``n`` equispaced nodes of ``[a, b)``, period ``b - a``.

    For even ``n`` the Nyquist mode is the symmetric one, ``cos(n θ / 2)``: its odd derivatives are taken as
    zero (it has no well-defined odd derivative at the nodes), the convention of the existing
    ``scheme="spectral"`` kernels.
    """

    family = "fourier"
    periodic = True
    _min_n = 2

    @functools.cached_property
    def nodes(self):
        return self.a + self.length * np.arange(self.n) / self.n

    @functools.cached_property
    def weights(self):
        return np.full(self.n, self.length / self.n)

    @functools.cached_property
    def wavenumbers(self) -> np.ndarray:
        """Angular wavenumbers in ``fftfreq`` order: ``2π/L · (0, 1, …, -1)``."""
        return 2.0 * np.pi * np.fft.fftfreq(self.n, d=self.length / self.n)

    def symbol(self, order: int, *, real: bool = False) -> np.ndarray:
        """``(i k)^order`` per mode, ``fftfreq`` order (``rfftfreq`` order when ``real``), Nyquist odd orders zeroed."""
        n = self.n
        k = 2.0 * np.pi * (np.fft.rfftfreq(n, d=self.length / n) if real else np.fft.fftfreq(n, d=self.length / n))
        s = (1j * k) ** int(order)
        if int(order) % 2 == 1 and n % 2 == 0:
            s[n // 2] = 0.0
        return s

    def diff(self, u, order: int = 1, axis: int = 0):
        u = _inexact(u)
        order = int(order)
        if order == 0:
            return u
        nd = u.ndim
        if jnp.iscomplexobj(u):
            s = jnp.asarray(self.symbol(order)).reshape(_shape_along(self.n, axis, nd))
            return jnp.fft.ifft(s * jnp.fft.fft(u, axis=axis), axis=axis)
        s = self.symbol(order, real=True)
        s = jnp.asarray(s).reshape(_shape_along(s.size, axis, nd))
        return jnp.fft.irfft(s * jnp.fft.rfft(u, axis=axis), n=self.n, axis=axis).astype(u.dtype)

    def interp_matrix(self, x):
        """The periodic cardinal functions (band-limited sinc kernel) at ``x``."""
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        n = self.n
        theta = 2.0 * np.pi * (x - self.nodes[None, :]) / self.length
        half = 0.5 * theta
        s = np.sin(half)
        num = np.sin(n * half)
        den = n * (np.tan(half) if n % 2 == 0 else s)
        hit = np.abs(s) < 1e-14
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(hit, 0.0, num / np.where(hit, 1.0, den))
        # a point on a node (an offset of whole spacings) takes the cardinal values exactly: 1 on its own node
        # (θ ≡ 0 mod 2π), 0 on every other one, instead of sin(π m) round-off
        t = (x - self.nodes[None, :]) * (n / self.length)
        on = np.abs(t - np.round(t)) < 1e-12
        rows = on.all(axis=1)
        if rows.any():
            out[rows] = (np.round(t[rows]).astype(int) % n == 0).astype(float)
        return np.where(hit, 1.0, out)

    def forward(self, u, axis: int = 0):
        """Complex coefficients ``ĉ_k`` in ``fftfreq`` order: ``u(x) = Σ ĉ_k e^{i k (x - a)}``."""
        return jnp.fft.fft(_inexact(u), axis=axis) / self.n

    def backward(self, c, axis: int = 0, *, real: bool = True):
        """Nodal values from :meth:`forward`'s coefficients; ``real`` drops the imaginary round-off."""
        u = jnp.fft.ifft(jnp.asarray(c) * self.n, axis=axis)
        return u.real if real else u

    def resample(self, u, n: int, axis: int = 0):
        """Values on ``n`` equispaced nodes of the same period, by zero-padding or truncating the spectrum.

        Padding is exact (the interpolant is unchanged); truncation is the L2 projection onto the smaller
        space, with the new Nyquist mode keeping its cosine part. O(n log n), unlike :meth:`to` (dense).
        """
        u = _inexact(u)
        n0, n = self.n, int(n)
        if n == n0:
            return u
        if jnp.iscomplexobj(u):
            return self._resample_complex(u, n, axis)
        c = jnp.moveaxis(jnp.fft.rfft(u, axis=axis), axis, 0) / n0
        m = n // 2 + 1
        if n > n0:
            out = jnp.zeros((m,) + c.shape[1:], dtype=c.dtype).at[: c.shape[0]].set(c)
            if n0 % 2 == 0:
                out = out.at[n0 // 2].multiply(0.5)  # the symmetric Nyquist splits into ±n0/2
        else:
            out = c[:m]
            if n % 2 == 0:
                out = out.at[n // 2].set(2.0 * out[n // 2].real)  # ±n/2 alias onto one cosine
        out = jnp.moveaxis(out, 0, axis) * n
        return jnp.fft.irfft(out, n=n, axis=axis).astype(u.dtype)

    def _resample_complex(self, u, n, axis):
        n0 = self.n
        c = jnp.moveaxis(jnp.fft.fft(u, axis=axis), axis, 0) / n0
        k0 = np.fft.fftfreq(n0, d=1.0 / n0).astype(int)
        k1 = np.fft.fftfreq(n, d=1.0 / n).astype(int)
        kmax = min(n0, n) // 2
        out = jnp.zeros((n,) + c.shape[1:], dtype=c.dtype)
        for src, k in enumerate(k0):
            if abs(k) > kmax:
                continue
            if abs(k) == kmax and (min(n0, n) % 2 == 0):
                # Nyquist of the smaller grid: split (padding) or fold (truncation)
                if n > n0:
                    for dst in np.flatnonzero(np.abs(k1) == kmax):
                        out = out.at[dst].add(0.5 * c[src])
                else:
                    out = out.at[int(np.flatnonzero(np.abs(k1) == kmax)[0])].add(c[src])
                continue
            out = out.at[int(np.flatnonzero(k1 == k)[0])].add(c[src])
        return jnp.fft.ifft(jnp.moveaxis(out, 0, axis) * n, axis=axis)

    def to(self, u, other, axis: int = 0):
        if isinstance(other, Fourier) and (other.a, other.b) == (self.a, self.b) and other.n != self.n:
            return self.resample(u, other.n, axis)
        return super().to(u, other, axis)


# ── polynomial bases (Chebyshev, Legendre) ──────────────────────────────────────────────────────────────


def _barycentric_weights(r):
    """``w_j ∝ 1/Π_{k≠j}(r_j - r_k)``, computed in logs (no under/overflow at large n), max-normalised."""
    d = r[:, None] - r[None, :]
    np.fill_diagonal(d, 1.0)
    logw = -np.sum(np.log(np.abs(d)), axis=1)
    sign = np.prod(np.sign(d), axis=1)
    return sign * np.exp(logw - logw.max())


def _legendre_table(x, n):
    """``P_0..P_{n-1}`` at ``x``: ``(len(x), n)``, by the three-term recurrence."""
    x = np.asarray(x, dtype=float)
    P = np.empty((x.size, n))
    P[:, 0] = 1.0
    if n > 1:
        P[:, 1] = x
    for k in range(1, n - 1):
        P[:, k + 1] = ((2 * k + 1) * x * P[:, k] - k * P[:, k - 1]) / (k + 1)
    return P


def _legendre_and_derivative(x, N):
    """``P_N(x)``, ``P_N'(x)``, ``P_N''(x)`` (the last two for |x| < 1)."""
    p0, p1 = np.ones_like(x), x.copy()
    if N == 0:
        return p0, np.zeros_like(x), np.zeros_like(x)
    for k in range(1, N):
        p0, p1 = p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
    dp = N * (x * p1 - p0) / (x * x - 1.0)
    ddp = (2 * x * dp - N * (N + 1) * p1) / (1.0 - x * x)
    return p1, dp, ddp


def _golub_welsch(off):
    """Nodes and normalised first-component weights of the symmetric tridiagonal Jacobi matrix (zero diagonal)."""
    from scipy.linalg import eigh_tridiagonal

    vals, vecs = eigh_tridiagonal(np.zeros(off.size + 1), off)
    return vals, vecs[0, :] ** 2


class _Polynomial(Axis):
    """Polynomial interpolation on a Gauss or Gauss-Lobatto node set of ``[a, b]`` (degree ``n - 1``)."""

    _min_n = 2
    #: below this size, derivatives and transforms are dense per-axis matmuls; above it a subclass may use a
    #: fast transform. Measured, see tests/test_spectral_bases.py and the plan file.
    dense_max = 1 << 30

    def __init__(self, n, a=-1.0, b=1.0, rule="lobatto"):
        if rule not in ("lobatto", "gauss"):
            raise ValueError(f"jno.spectral: rule={rule!r}; use 'lobatto' (ends included) or 'gauss' (interior).")
        self.rule = rule
        if rule == "gauss":
            self._min_n = 1
        super().__init__(n, a, b)

    def _extra(self):
        return {"rule": self.rule}

    @property
    def scale(self) -> float:
        """``d r / d x`` of the map from ``[a, b]`` to the reference ``[-1, 1]``."""
        return 2.0 / self.length

    def _to_ref(self, x):
        return (np.asarray(x, dtype=float) - self.a) * self.scale - 1.0

    @functools.cached_property
    def ref_nodes(self) -> np.ndarray:
        raise NotImplementedError

    @functools.cached_property
    def ref_weights(self) -> np.ndarray:
        raise NotImplementedError

    @functools.cached_property
    def nodes(self):
        return self.a + (self.ref_nodes + 1.0) / self.scale

    @functools.cached_property
    def weights(self):
        return self.ref_weights / self.scale

    @functools.cached_property
    def bary(self) -> np.ndarray:
        return _barycentric_weights(self.ref_nodes)

    @functools.cached_property
    def _d1_ref(self):
        r, w = self.ref_nodes, self.bary
        d = r[:, None] - r[None, :]
        np.fill_diagonal(d, 1.0)
        D = (w[None, :] / w[:, None]) / d
        np.fill_diagonal(D, 0.0)
        np.fill_diagonal(D, -D.sum(axis=1))
        return D

    def _diff_matrix(self, order):
        cache = self.__dict__.setdefault("_dcache", {})
        if order not in cache:
            cache[order] = np.eye(self.n) if order == 0 else np.linalg.matrix_power(self._d1_ref * self.scale, order)
        return cache[order]

    def diff(self, u, order: int = 1, axis: int = 0):
        order = int(order)
        if order == 0:
            return _inexact(u)
        return apply_axis(self._diff_matrix(order), u, axis)

    def interp_matrix(self, x):
        """Barycentric formula of the second kind; a point on a node gets that node's cardinal value exactly."""
        r = self._to_ref(x).reshape(-1, 1)
        d = r - self.ref_nodes[None, :]
        hit = np.abs(d) < 1e-14
        with np.errstate(divide="ignore", invalid="ignore"):
            t = self.bary[None, :] / np.where(hit, 1.0, d)
        t = np.where(hit, 0.0, t)
        L = t / t.sum(axis=1, keepdims=True)
        rows = hit.any(axis=1)
        if rows.any():
            L[rows] = hit[rows].astype(float)
        return L

    # modal transforms through the Vandermonde matrix of the family's polynomials
    def _vandermonde(self, r) -> np.ndarray:
        raise NotImplementedError

    @functools.cached_property
    def _V(self):
        return self._vandermonde(self.ref_nodes)

    @functools.cached_property
    def _Vinv(self):
        return np.linalg.inv(self._V)

    def forward(self, u, axis: int = 0):
        """Modal coefficients ``c_k`` of the interpolant: ``u = Σ c_k φ_k(r)``, ``φ_k`` the family's polynomials."""
        return apply_axis(self._Vinv, u, axis)

    def backward(self, c, axis: int = 0, **_):
        return apply_axis(self._V, c, axis)

    def modal_eval(self, c, x, axis: int = 0):
        """Evaluate the modal series ``Σ c_k φ_k`` at ``x``."""
        return apply_axis(self._vandermonde(self._to_ref(np.asarray(x, dtype=float).reshape(-1))), c, axis)


class Chebyshev(_Polynomial):
    """Chebyshev interpolation. ``rule="lobatto"``: ``r_j = -cos(π j / (n-1))`` (ends included, the usual
    collocation set); ``rule="gauss"``: ``r_j = -cos(π (2j+1) / (2n))`` (interior only). Coefficients are in the
    Chebyshev polynomials ``T_k``. Above :attr:`dense_max` nodes the Lobatto transforms and derivatives run by
    FFT (the discrete cosine transform of type I as an FFT of the even extension) instead of dense matmuls."""

    family = "chebyshev"
    dense_max = 256  # measured 2026-11-27: dense wins below on the 3070 (f64) and up to 513+ on the 13600K

    @functools.cached_property
    def ref_nodes(self):
        n = self.n
        if self.rule == "lobatto":
            return -np.cos(np.pi * np.arange(n) / (n - 1))
        return -np.cos(np.pi * (2 * np.arange(n) + 1) / (2 * n))

    @functools.cached_property
    def ref_weights(self):
        n = self.n
        if self.rule == "lobatto":
            # Clenshaw-Curtis (Trefethen, Spectral Methods in MATLAB, clencurt.m), ascending order
            N = n - 1
            theta = np.pi * np.arange(n) / N
            w = np.zeros(n)
            ii = np.arange(1, N)
            v = np.ones(N - 1)
            if N % 2 == 0:
                w[0] = w[N] = 1.0 / (N * N - 1)
                for k in range(1, N // 2):
                    v -= 2.0 * np.cos(2 * k * theta[ii]) / (4 * k * k - 1)
                v -= np.cos(N * theta[ii]) / (N * N - 1)
            else:
                w[0] = w[N] = 1.0 / (N * N)
                for k in range(1, (N - 1) // 2 + 1):
                    v -= 2.0 * np.cos(2 * k * theta[ii]) / (4 * k * k - 1)
            w[ii] = 2.0 * v / N
            return w[::-1].copy()
        # Fejér's first rule on the Chebyshev-Gauss nodes
        theta = np.pi * (2 * np.arange(n) + 1) / (2 * n)
        k = np.arange(1, n // 2 + 1)
        s = np.cos(2 * np.outer(theta, k)) / (4 * k * k - 1)
        w = (2.0 / n) * (1.0 - 2.0 * s.sum(axis=1))
        return w[::-1].copy()

    def _vandermonde(self, r):
        r = np.clip(np.asarray(r, dtype=float), -1.0, 1.0)
        return np.cos(np.outer(np.arccos(r), np.arange(self.n)))

    # FFT path (Lobatto only) ---------------------------------------------------------------------------
    def _fast(self):
        return self.rule == "lobatto" and self.n > self.dense_max

    def forward(self, u, axis: int = 0):
        if not self._fast():
            return super().forward(u, axis)
        u = _inexact(u)
        if jnp.iscomplexobj(u):
            return self.forward(u.real, axis) + 1j * self.forward(u.imag, axis)
        N = self.n - 1
        v = jnp.flip(jnp.moveaxis(u, axis, 0), axis=0)  # descending r: v_j = u(cos(π j / N))
        ext = jnp.concatenate([v, jnp.flip(v[1:N], axis=0)], axis=0)  # even extension, length 2N
        c = jnp.fft.rfft(ext, axis=0).real / N
        c = c.at[0].multiply(0.5).at[N].multiply(0.5)
        return jnp.moveaxis(c, 0, axis)

    def backward(self, c, axis: int = 0, **_):
        if not self._fast():
            return super().backward(c, axis)
        c = _inexact(c)
        N = self.n - 1
        X = jnp.moveaxis(c, axis, 0) * N
        X = X.at[0].multiply(2.0).at[N].multiply(2.0)
        if jnp.iscomplexobj(X):
            v = jnp.fft.irfft(X.real, n=2 * N, axis=0) + 1j * jnp.fft.irfft(X.imag, n=2 * N, axis=0)
        else:
            v = jnp.fft.irfft(X, n=2 * N, axis=0)
        return jnp.moveaxis(jnp.flip(v[: N + 1], axis=0), 0, axis)

    @staticmethod
    def coeff_derivative(c, axis: int = 0):
        """Coefficients of ``d/dr`` of ``Σ c_k T_k(r)``: ``c'_k = (2/ĉ_k) Σ_{p>k, p+k odd} p c_p`` (ĉ_0 = 2)."""
        c = jnp.moveaxis(_inexact(c), axis, 0)
        n = c.shape[0]
        p = jnp.arange(n).reshape((n,) + (1,) * (c.ndim - 1))
        a = p * c
        even = p % 2 == 0
        # reversed cumulative sums over even and odd p
        s_even = jnp.flip(jnp.cumsum(jnp.flip(jnp.where(even, a, 0), axis=0), axis=0), axis=0)
        s_odd = jnp.flip(jnp.cumsum(jnp.flip(jnp.where(even, 0, a), axis=0), axis=0), axis=0)
        d = 2.0 * jnp.where(even, s_odd, s_even)  # k even needs odd p > k; k odd needs even p > k
        d = d.at[0].multiply(0.5).at[n - 1].set(0.0)
        return jnp.moveaxis(d, 0, axis)

    def diff(self, u, order: int = 1, axis: int = 0):
        order = int(order)
        if order == 0 or not self._fast():
            return super().diff(u, order, axis)
        c = self.forward(u, axis)
        for _ in range(order):
            c = self.coeff_derivative(c, axis) * self.scale
        return self.backward(c, axis)


class Legendre(_Polynomial):
    """Legendre interpolation. ``rule="lobatto"``: the Gauss-Lobatto-Legendre (GLL) nodes, the zeros of
    ``(1 - r²) P'_{n-1}``; ``rule="gauss"``: the Gauss-Legendre nodes, the zeros of ``P_n``. Coefficients are in
    the Legendre polynomials ``P_k``. Nodes by Golub-Welsch, polished by one Newton step."""

    family = "legendre"

    @functools.cached_property
    def _rule(self):
        n = self.n
        if self.rule == "gauss":
            k = np.arange(1, n)
            r, w0 = _golub_welsch(k / np.sqrt(4.0 * k * k - 1.0))
            p, dp, _ = _legendre_and_derivative(r, n)
            r = r - p / dp
            p, dp, _ = _legendre_and_derivative(r, n)
            w = 2.0 / ((1.0 - r * r) * dp * dp)
            return r, w
        N = n - 1
        if N == 1:
            return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
        # interior GLL nodes = Gauss-Jacobi(1,1) nodes of degree N-1 (Gegenbauer λ = 3/2)
        k = np.arange(1, N - 1)
        off = np.sqrt(k * (k + 2.0) / ((2 * k + 1.0) * (2 * k + 3.0)))
        inner = _golub_welsch(off)[0] if off.size else np.array([0.0])
        _, dp, ddp = _legendre_and_derivative(inner, N)
        inner = inner - dp / ddp
        r = np.concatenate([[-1.0], np.sort(inner), [1.0]])
        pN = _legendre_table(r, N + 1)[:, N]
        w = 2.0 / (N * (N + 1) * pN * pN)
        return r, w

    @functools.cached_property
    def ref_nodes(self):
        return self._rule[0]

    @functools.cached_property
    def ref_weights(self):
        return self._rule[1]

    def _vandermonde(self, r):
        return _legendre_table(np.asarray(r, dtype=float), self.n)


_FAMILIES = {"fourier": Fourier, "chebyshev": Chebyshev, "legendre": Legendre}


def axis_from_spec(family: str, n: int, a: float, b: float, rule: str | None = None) -> Axis:
    """Build an axis by family name (``"fourier"``, ``"chebyshev"``, ``"legendre"``)."""
    fam = str(family).lower()
    if fam not in _FAMILIES:
        raise ValueError(f"jno.spectral: unknown basis family {family!r}; use one of {sorted(_FAMILIES)}.")
    cls = _FAMILIES[fam]
    if cls is Fourier:
        if rule not in (None, "uniform"):
            raise ValueError(f"jno.spectral: a Fourier axis has equispaced nodes only, got rule={rule!r}.")
        return Fourier(n, a, b)
    return cls(n, a, b, rule=rule or "lobatto")
