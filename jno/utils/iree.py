import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import jax
from jax import export
import iree.runtime as ireert

log = logging.getLogger(__name__)


class IREEModel:
    """Compiled IREE model — serialisable via ``jno.save`` / ``jno.load``.

    Typical usage::

        # Compile once
        model = IREEModel.compile(my_jax_fn, sample_inputs)

        # Save / load using the standard jno API
        jno.save(model, "model.pkl")
        model = jno.load("model.pkl")

        output = model(x, y)

    The IREE runtime objects (context, module handle) are not picklable
    and are reconstructed transparently on unpickle via ``__setstate__``.
    """

    def __init__(self, vmfb_bytes: bytes, module_name: str, device: str = "local-sync"):
        self.vmfb_bytes = vmfb_bytes
        self.module_name = module_name
        self.device = device
        # Runtime state — rebuilt in _load_model and after unpickling
        self._tmpfile_path: str | None = None
        self._ctx = None
        self._module = None
        # Expose a logger so jno.save(model, path) works out of the box
        self.log = log
        self._load_model()

    # ── IREE runtime setup ────────────────────────────────────────────────────

    def _load_model(self):
        """Write vmfb bytes to a temp file and mmap into IREE."""
        # iree-runtime's mmap requires a real file path, so we write once.
        fd, path = tempfile.mkstemp(suffix=".vmfb")
        try:
            os.write(fd, self.vmfb_bytes)
        finally:
            os.close(fd)
        self._tmpfile_path = path

        config = ireert.Config(self.device)
        self._ctx = ireert.SystemContext(config=config)
        vm_module = ireert.VmModule.mmap(self._ctx.instance, path)
        self._ctx.add_vm_module(vm_module)
        self._module = self._ctx.modules[self.module_name]

    def __del__(self):
        if self._tmpfile_path and os.path.exists(self._tmpfile_path):
            try:
                os.unlink(self._tmpfile_path)
            except OSError:
                pass

    # ── Pickle support ────────────────────────────────────────────────────────

    def __getstate__(self):
        """Only serialise the data — IREE handles are rebuilt on load."""
        return {
            "vmfb_bytes": self.vmfb_bytes,
            "module_name": self.module_name,
            "device": self.device,
        }

    def __setstate__(self, state):
        self.vmfb_bytes = state["vmfb_bytes"]
        self.module_name = state["module_name"]
        self.device = state["device"]
        self._tmpfile_path = None
        self._ctx = None
        self._module = None
        self.log = log
        self._load_model()

    # ── Inference ─────────────────────────────────────────────────────────────

    def __call__(self, *args) -> np.ndarray:
        """Run inference and return a numpy array."""
        return np.asarray(self._module.main(*args))  # type: ignore[attr-defined]

    def infer_raw(self, *args):
        """Run inference and return an IREE DeviceArray (no host copy)."""
        return self._module.main(*args)  # type: ignore[attr-defined]

    # ── Compile ───────────────────────────────────────────────────────────────

    @classmethod
    def compile(
        cls,
        func: Callable,
        sample_inputs: tuple,
        target_backend: str = "llvm-cpu",
        optimization_level: int = 3,
    ) -> "IREEModel":
        """Compile a JAX function to an :class:`IREEModel`.

        The returned object can be persisted with ``jno.save`` / ``jno.load``::

            model = IREEModel.compile(infer, sample_inputs)
            jno.save(model, "model.pkl")

        Args:
            func: JAX-compatible function to compile.
            sample_inputs: Example inputs (used for shape inference).
            target_backend: IREE target backend (default ``"llvm-cpu"``).
            optimization_level: IREE optimisation level 0–3 (default ``3``).

        Returns:
            A ready-to-use :class:`IREEModel` instance.
        """
        # 1. Export JAX → MLIR bytecode
        exported = export.export(jax.jit(func))(*sample_inputs)

        mlir_tmp = vmfb_tmp = None
        try:
            # 2. Write MLIR to a temp file
            fd, mlir_tmp = tempfile.mkstemp(suffix=".mlir")
            os.write(fd, exported.mlir_module_serialized)
            os.close(fd)

            # 3. Compile MLIR → vmfb via iree-compile
            fd, vmfb_tmp = tempfile.mkstemp(suffix=".vmfb")
            os.close(fd)

            cmd = [
                "iree-compile",
                mlir_tmp,
                f"--iree-hal-target-backends={target_backend}",
                f"--iree-opt-level=O{optimization_level}",
                "-o",
                vmfb_tmp,
            ]
            if target_backend == "llvm-cpu":
                cmd.extend(
                    [
                        "--iree-llvmcpu-target-cpu=host",
                        "--iree-llvmcpu-target-cpu-features=host",
                        "--iree-llvmcpu-enable-ukernels=all",
                        "--iree-llvmcpu-stack-allocation-limit=524288",
                        "--iree-llvmcpu-loop-vectorization",
                        "--iree-llvmcpu-loop-unrolling",
                        "--iree-llvmcpu-loop-interleaving",
                        "--iree-llvmcpu-slp-vectorization",
                        "--iree-opt-data-tiling",
                        "--iree-opt-const-eval",
                        "--iree-opt-strip-assertions",
                    ]
                )

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                log.error("iree-compile failed:\n%s", e.stderr)
                raise

            # 4. Read compiled bytes
            vmfb_bytes = Path(vmfb_tmp).read_bytes()

        finally:
            for p in (mlir_tmp, vmfb_tmp):
                if p and os.path.exists(p):
                    os.unlink(p)

        module_name = f"jit_{func.__name__}"
        log.info("Compiled IREE model  (module: %s, %d KB)", module_name, len(vmfb_bytes) // 1024)
        return cls(vmfb_bytes, module_name=module_name)
