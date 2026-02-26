import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import jax
from jax import export
import iree.runtime as ireert


class IREEModel:
    """Wrapper for loading and running IREE models"""

    def __init__(self, vmfb_path: str, module_name: str = "jit_infer", device: str = "local-sync"):
        self.vmfb_path = vmfb_path
        self.module_name = module_name
        self.device = device
        self._ctx = None
        self._module = None
        self._load_model()

    def _load_model(self):
        config = ireert.Config(self.device)
        self._ctx = ireert.SystemContext(config=config)
        vm_module = ireert.VmModule.mmap(self._ctx.instance, self.vmfb_path)
        self._ctx.add_vm_module(vm_module)
        self._module = self._ctx.modules[self.module_name]

    def __call__(self, *args) -> np.ndarray:
        """Run inference and return numpy array."""
        result = self._module.main(*args)
        return np.asarray(result)

    def infer_raw(self, *args):
        """Run inference and return IREE DeviceArray (no host copy)."""
        return self._module.main(*args)

    @staticmethod
    def save(func: Callable, sample_inputs: tuple, output_path: str = "model.vmfb", target_backend: str = "llvm-cpu", optimization_level: int = 3) -> "IREEModel":
        """Export a JAX function to IREE VMFB format."""
        print(ireert.query_available_drivers())
        output_path = Path(output_path)

        # Export from JAX
        exported = export.export(jax.jit(func))(*sample_inputs)

        with tempfile.NamedTemporaryFile(suffix=".mlir", delete=False) as f:
            mlir_path = f.name
            f.write(exported.mlir_module_serialized)

        # Build compile command
        cmd = [
            "iree-compile",
            mlir_path,
            f"--iree-hal-target-backends={target_backend}",
            f"--iree-opt-level=O{optimization_level}",
            "-o",
            str(output_path),
        ]

        if target_backend == "llvm-cpu":
            cmd.extend(
                [
                    "--iree-llvmcpu-target-cpu=host",
                    "--iree-llvmcpu-target-cpu-features=host",
                    "--iree-llvmcpu-enable-ukernels=all",
                    "--iree-llvmcpu-stack-allocation-limit=524288",  # 512KB
                    # Enable microkernels
                    "--iree-llvmcpu-enable-ukernels=all",
                    # LLVM optimizations
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
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Compilation failed:\n{e.stderr}")
            raise
        finally:
            Path(mlir_path).unlink()

        module_name = f"jit_{func.__name__}"
        print(f"The name of the module is {module_name}")
        return IREEModel(str(output_path), module_name=module_name)

    @staticmethod
    def load(path: str = "model.vmfb", module_name: str = "jit_infer") -> "IREEModel":
        return IREEModel(str(path), module_name)
