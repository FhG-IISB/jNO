"""Differentiable, GPU-resident symmetric direct solve: cuDSS LDL^T as XLA custom calls.

``solve(plan, values, b)`` factors the symmetric matrix -- real symmetric indefinite, or complex SYMMETRIC
(A^T = A, not Hermitian: an eddy-current operator nu curl curl + j w sigma) -- whose (COO) values are the device
array ``values``
and solves ``A x = b`` for one or several right-hand sides, all inside the compiled program: no host
callback, so ``jax.jit(jax.value_and_grad(f))`` runs the assembly, the factorisation, the solves and the
adjoint on the GPU. Reverse mode goes through :func:`jax.lax.custom_linear_solve` (A^T = A, so the adjoint
is the same solve with the cotangent as right-hand side, on the SAME factors); the cotangent of ``values``
is -lambda_i x_j on the pattern, plain JAX.

    plan = cudss_sym.plan(rows, cols, n)          # once: the sparsity (numpy, concrete), registers a slot
    plan = cudss_sym.plan(rows, cols, n, complex_values=True)   # complex symmetric: half the unknowns of the
                                                  # real-equivalent 2n block, ~half its factor flops and memory
    x = cudss_sym.solve(plan, values, b)          # under jit / grad; b (n,) or (n, k)
    x = cudss_sym.solve(plan, values, b, refactor=False)   # REUSE the slot's last factors (of a nearby matrix) as a
                                                  # preconditioner: GMRES to `tol`, NaN if it fails

Rules of the design:
  * the pattern is CONCRETE and fixed per plan (a mesh with fixed topology); values change every call;
  * the matrix must be exactly symmetric (both triangles present; the upper one is used);
  * ONE factorisation per slot per program execution: two ``solve`` calls on one plan inside one program
    (a vmap over designs, a scan) would refactor the same slot between a forward and its adjoint. Use one
    plan per concurrent matrix.
  * no ``vmap`` rule: batching raises.
  * ``refactor=False`` (an optimisation loop whose matrix moves a little per step): no factorisation; restarted
    GMRES preconditioned by the slot's stale LDL^T, until every column's relative residual is below ``tol``. If it
    has not converged within ``maxiter`` preconditioner solves, the result (and the gradient) is NaN: refactor.

The extension is compiled on first use with ``nvcc`` against the cuDSS of the ``nvidia-cudss`` wheel and JAX's
FFI headers (cached by source hash). :func:`available` is False without a GPU, nvcc or cuDSS.
"""

from __future__ import annotations

import ctypes
import hashlib
import itertools
import os
import subprocess
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "_ext" / "cudss_sym_ffi.cu"
_LIB = None
_TRIED = False
_SLOTS = itertools.count(1)


def _nvcc():
    for c in (os.environ.get("NVCC"), "nvcc", "/opt/cuda/bin/nvcc", "/usr/local/cuda/bin/nvcc"):
        if c and (Path(c).exists() or subprocess.run(["which", c], capture_output=True).returncode == 0):
            return c
    return None


def _load():
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True
    try:
        import jax

        site = Path(jax.__file__).resolve().parents[1]
        inc_jax, nv = site / "jaxlib" / "include", site / "nvidia" / "cu12"
        if jax.default_backend() not in ("gpu", "cuda") or not _SRC.exists():
            return None
        if not (nv / "include" / "cudss.h").exists() or not (nv / "lib" / "libcudss.so.0").exists():
            return None
        key = hashlib.sha256(_SRC.read_bytes() + str(inc_jax).encode()).hexdigest()[:16]
        cache = Path(os.environ.get("JNO_EXT_CACHE", Path.home() / ".cache" / "jno" / "ext"))
        so = cache / f"libjno_cudss_sym_{key}.so"
        if not so.exists():
            nvcc = _nvcc()
            if nvcc is None:
                return None
            cache.mkdir(parents=True, exist_ok=True)
            tmp = so.with_suffix(f".{os.getpid()}.tmp")
            cmd = [nvcc, "-O2", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", f"-I{inc_jax}", f"-I{nv / 'include'}",
                   str(_SRC), "-o", str(tmp), f"-L{nv / 'lib'}", "-l:libcudss.so.0", "-Xlinker", f"-rpath={nv / 'lib'}"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError("nvcc failed:\n" + r.stderr[-3000:])
            tmp.replace(so)
        lib = ctypes.cdll.LoadLibrary(str(so))
        lib.jno_cudsd_create.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.jno_cudsd_create.restype = ctypes.c_int
        lib.jno_cudsd_free.argtypes = [ctypes.c_int64]
        lib.jno_cudsd_release.argtypes = [ctypes.c_int64]
        lib.jno_cudsd_release.restype = ctypes.c_int
        lib.jno_cudsd_stats.argtypes = [ctypes.c_int64, ctypes.c_void_p]
        lib.jno_cudsd_stats.restype = ctypes.c_int
        for name in ("jno_cudsd_factor", "jno_cudsd_solve", "jno_cudsz_factor", "jno_cudsz_solve"):
            jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(getattr(lib, name)), platform="CUDA")
        _LIB = lib
    except RuntimeError:
        raise
    except Exception:  # noqa: BLE001 -- optional: any other failure means "not available"
        _LIB = None
    return _LIB


def available() -> bool:
    return _load() is not None


class Plan:
    """The fixed sparsity of one symmetric matrix and the device slot that holds its factors."""

    def __init__(self, rows, cols, n, complex_values=False):
        lib = _load()
        if lib is None:
            raise RuntimeError("jno: the cuDSS symmetric solve is not available (needs a GPU, nvcc and nvidia-cudss).")
        import jax.numpy as jnp

        rows, cols = np.asarray(rows, np.int64), np.asarray(cols, np.int64)
        self.n, self.nse = int(n), len(rows)
        up = cols >= rows
        key = rows[up] * self.n + cols[up]
        uniq, inv = np.unique(key, return_inverse=True)
        self.nnz = len(uniq)
        pos = np.full(self.nse, self.nnz, np.int32)      # entries below the diagonal go to a dropped dummy slot
        pos[up] = inv.astype(np.int32)
        ptr = np.searchsorted(uniq // self.n, np.arange(self.n + 1)).astype(np.int32)
        idx = (uniq % self.n).astype(np.int32)
        self.complex = bool(complex_values)
        self.slot = next(_SLOTS)
        st = lib.jno_cudsd_create(self.slot, self.n, self.nnz, ptr.ctypes.data, idx.ctypes.data, int(self.complex))
        if st != 0:
            raise RuntimeError(f"jno: cuDSS slot creation failed (status {st}).")
        self.pos = jnp.asarray(pos)
        self.indices = jnp.asarray(np.stack([rows, cols], 1).astype(np.int32))

    def stats(self) -> dict:
        """After a factorisation: nnz of the factors, perturbed pivots, cuDSS's device-memory estimates (bytes)."""
        out = np.zeros(4, np.int64)
        if _load().jno_cudsd_stats(self.slot, out.ctypes.data) != 0:
            return {}
        return {"lu_nnz": int(out[0]), "npivots": int(out[1]), "device_permanent": int(out[2]), "device_peak": int(out[3])}

    def release(self) -> None:
        """Free this slot's factors (cuDSS analysis + factor memory) between programs; the sparsity and the slot
        id stay, so a compiled program that uses the slot again re-runs the analysis and works unchanged. Call it
        only when no program using the slot is running (it synchronises the device)."""
        st = _load().jno_cudsd_release(self.slot)
        if st != 0:
            raise RuntimeError(f"jno: cuDSS slot release failed (status {st}).")

    def __del__(self):
        try:
            lib = _load()
            if lib is not None:
                lib.jno_cudsd_free(self.slot)
        except Exception:  # noqa: BLE001
            pass


def plan(rows, cols, n, complex_values=False) -> Plan:
    return Plan(rows, cols, n, complex_values)


def _ffi_solve(p: Plan, token, rhs):
    """M^-1 rhs on the slot's current factors; rhs (n,) or (n, k)."""
    import jax

    sol = "jno_cudsz_solve" if p.complex else "jno_cudsd_solve"
    r2 = rhs.reshape(p.n, -1).T                             # (k, n): each right-hand side contiguous
    x = jax.ffi.ffi_call(sol, jax.ShapeDtypeStruct(r2.shape, r2.dtype))(token, r2, slot=np.int64(p.slot))
    return x.T.reshape(rhs.shape)


def _refine(p: Plan, matvec, rhs, tol, maxiter, restart=10):
    """Restarted GLOBAL GMRES (one Krylov space for all columns, Frobenius inner product), right-preconditioned by
    the slot's (stale) factors M: x = M^-1 y, A M^-1 y = b. Stops when every column's TRUE relative residual is
    below ``tol``. Returns (x, preconditioner solves, worst relative residual). Memory: restart + 1 copies of rhs."""
    import jax
    import jax.numpy as jnp

    token = jnp.zeros((1,), jnp.int32)                      # no factorisation in this program: no dependency
    prec = lambda r: _ffi_solve(p, token, r)
    B = rhs.reshape(p.n, -1)
    bn = jnp.maximum(jnp.linalg.norm(B, axis=0), jnp.finfo(jnp.float64).tiny)
    relres = lambda R: jnp.max(jnp.linalg.norm(R, axis=0) / bn)
    target = 0.5 * tol * jnp.min(bn)                        # ||R||_F below this -> every column below tol
    mv = lambda X: matvec(X.reshape(rhs.shape)).reshape(B.shape)
    pc = lambda X: prec(X.reshape(rhs.shape)).reshape(B.shape)
    m, dot = restart, lambda U, W: jnp.vdot(U, W)

    def cycle(state):
        X, R, nsolve, _ = state
        beta = jnp.linalg.norm(R)
        V = jnp.zeros((m + 1,) + B.shape, B.dtype).at[0].set(R / beta)
        H = jnp.zeros((m + 1, m), B.dtype)
        e1 = jnp.zeros(m + 1, B.dtype).at[0].set(beta)

        def arnoldi(st):
            V, H, k, _ = st
            W = mv(pc(V[k]))
            mask = (jnp.arange(m + 1) <= k).astype(B.dtype)
            for _ in range(2):                              # classical Gram-Schmidt, twice
                h = jnp.einsum("jnk,nk->j", V.conj(), W) * mask
                W = W - jnp.einsum("j,jnk->nk", h, V)
                H = H.at[:, k].add(h)
            hn = jnp.linalg.norm(W)
            H = H.at[k + 1, k].set(hn)
            V = V.at[k + 1].set(W / jnp.where(hn > 0, hn, 1.0))
            y = jnp.linalg.lstsq(H, e1)[0]
            return V, H, k + 1, jnp.linalg.norm(e1 - H @ y)

        V, H, k, _ = jax.lax.while_loop(lambda st: (st[2] < m) & (st[3] > target), arnoldi, (V, H, 0, beta))
        y = jnp.linalg.lstsq(H, e1)[0]
        X = X + pc(jnp.einsum("j,jnk->nk", y[:m], V[:m]))
        R = B - mv(X)
        return X, R, nsolve + k + 1, relres(R)

    X0 = jnp.zeros_like(B)
    X, _R, it, rel = jax.lax.while_loop(lambda s: (s[3] > tol) & (s[2] < maxiter), cycle, (X0, B, 0, jnp.inf))
    return X.reshape(rhs.shape), it, rel


def stale_solve(p: Plan, values, b, tol=1e-10, maxiter=30):
    """The iteration of ``solve(..., refactor=False)`` without autodiff, with its diagnostics:
    (x, preconditioner solves, worst relative residual). Not differentiable; for choosing ``tol`` and when to refactor."""
    import jax.experimental.sparse as jsp
    import jax.numpy as jnp

    dt = jnp.complex128 if p.complex else jnp.float64
    A = jsp.BCOO((values.astype(dt), p.indices), shape=(p.n, p.n))
    return _refine(p, lambda x: A @ x, b.astype(dt), tol, maxiter)


def solve(p: Plan, values, b, refactor=True, tol=1e-10, maxiter=30):
    """``x = A^{-1} b`` for the symmetric matrix with COO ``(p.indices, values)``; jit- and grad-able.

    ``refactor=False`` reuses the slot's last factors (see the module notes): NaN where GMRES has not
    reached ``tol`` within ``maxiter`` steps, in the solution and in the gradient."""
    import jax
    import jax.experimental.sparse as jsp
    import jax.numpy as jnp

    n, slot = p.n, np.int64(p.slot)
    dt = jnp.complex128 if p.complex else jnp.float64
    fac = "jno_cudsz_factor" if p.complex else "jno_cudsd_factor"
    values = values.astype(dt)
    b = b.astype(dt)
    A = jsp.BCOO((values, p.indices), shape=(n, n))
    if refactor:
        vcsr = jnp.zeros(p.nnz + 1, dt).at[p.pos].add(values)[: p.nnz]
        # the factorisation is not differentiated: the cotangent of `values` enters through the matvec below
        token = jax.ffi.ffi_call(fac, jax.ShapeDtypeStruct((1,), jnp.int32))(jax.lax.stop_gradient(vcsr), slot=slot)
        _solve = lambda _matvec, rhs: _ffi_solve(p, token, rhs)
    else:
        def _solve(matvec, rhs):
            x, _it, rel = _refine(p, matvec, rhs, tol, maxiter)
            return jnp.where(rel <= tol, x, jnp.nan)

    # symmetric=True: the adjoint is the plain TRANSPOSE (JAX's linear transpose does not conjugate), and A^T = A
    # holds for complex symmetric as for real symmetric -- the same solve on the same factors
    return jax.lax.custom_linear_solve(lambda x: A @ x, b, _solve, symmetric=True)
