"""Differentiable, GPU-resident symmetric direct solve: cuDSS LDL^T as XLA custom calls.

``solve(plan, values, b)`` factors the symmetric matrix whose (COO) values are the device array ``values``
and solves ``A x = b`` for one or several right-hand sides, all inside the compiled program: no host
callback, so ``jax.jit(jax.value_and_grad(f))`` runs the assembly, the factorisation, the solves and the
adjoint on the GPU. Reverse mode goes through :func:`jax.lax.custom_linear_solve` (A^T = A, so the adjoint
is the same solve with the cotangent as right-hand side, on the SAME factors); the cotangent of ``values``
is -lambda_i x_j on the pattern, plain JAX.

    plan = cudss_sym.plan(rows, cols, n)          # once: the sparsity (numpy, concrete), registers a slot
    x = cudss_sym.solve(plan, values, b)          # under jit / grad; b (n,) or (n, k)

Rules of the design:
  * the pattern is CONCRETE and fixed per plan (a mesh with fixed topology); values change every call;
  * the matrix must be exactly symmetric (both triangles present; the upper one is used);
  * ONE factorisation per slot per program execution: two ``solve`` calls on one plan inside one program
    (a vmap over designs, a scan) would refactor the same slot between a forward and its adjoint. Use one
    plan per concurrent matrix.
  * no ``vmap`` rule: batching raises.

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
        lib.jno_cudsd_create.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p]
        lib.jno_cudsd_create.restype = ctypes.c_int
        lib.jno_cudsd_free.argtypes = [ctypes.c_int64]
        jax.ffi.register_ffi_target("jno_cudsd_factor", jax.ffi.pycapsule(lib.jno_cudsd_factor), platform="CUDA")
        jax.ffi.register_ffi_target("jno_cudsd_solve", jax.ffi.pycapsule(lib.jno_cudsd_solve), platform="CUDA")
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

    def __init__(self, rows, cols, n):
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
        self.slot = next(_SLOTS)
        st = lib.jno_cudsd_create(self.slot, self.n, self.nnz, ptr.ctypes.data, idx.ctypes.data)
        if st != 0:
            raise RuntimeError(f"jno: cuDSS slot creation failed (status {st}).")
        self.pos = jnp.asarray(pos)
        self.indices = jnp.asarray(np.stack([rows, cols], 1).astype(np.int32))

    def __del__(self):
        try:
            lib = _load()
            if lib is not None:
                lib.jno_cudsd_free(self.slot)
        except Exception:  # noqa: BLE001
            pass


def plan(rows, cols, n) -> Plan:
    return Plan(rows, cols, n)


def solve(p: Plan, values, b):
    """``x = A^{-1} b`` for the symmetric matrix with COO ``(p.indices, values)``; jit- and grad-able."""
    import jax
    import jax.experimental.sparse as jsp
    import jax.numpy as jnp

    n, slot = p.n, np.int64(p.slot)
    vcsr = jnp.zeros(p.nnz + 1, values.dtype).at[p.pos].add(values)[: p.nnz]
    # the factorisation is not differentiated: the cotangent of `values` enters through the matvec below
    token = jax.ffi.ffi_call("jno_cudsd_factor", jax.ShapeDtypeStruct((1,), jnp.int32))(
        jax.lax.stop_gradient(vcsr), slot=slot
    )
    A = jsp.BCOO((values, p.indices), shape=(n, n))

    def _solve(_matvec, rhs):
        r2 = rhs.reshape(n, -1).T                           # (k, n): each right-hand side contiguous
        x = jax.ffi.ffi_call("jno_cudsd_solve", jax.ShapeDtypeStruct(r2.shape, r2.dtype))(token, r2, slot=slot)
        return x.T.reshape(rhs.shape)

    return jax.lax.custom_linear_solve(lambda x: A @ x, b, _solve, symmetric=True)
