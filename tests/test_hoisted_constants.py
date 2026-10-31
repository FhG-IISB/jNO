"""A program's large constants reach it as arguments (`JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS`), in either import order:
the lowered program of a function closing over a large array carries no literal of it. `JNO_HOIST_CONSTANTS=0`, or
that setting given for JAX itself, keeps them embedded. Each case is a fresh process: JAX reads the setting when it is
first imported."""

import os
import subprocess
import sys

PROBE = """
import numpy as np
{imports}
big = jax.numpy.asarray(np.arange(1000.0) + 0.123456)
text = jax.jit(lambda x: x * big).lower(1.0).as_text()
print("EMBEDDED" if 'dense<"0x' in text else "HOISTED")
"""


def _lowered(imports, **extra_env):
    env = {k: v for k, v in os.environ.items() if k not in ("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "JNO_HOIST_CONSTANTS")}
    env.update(JAX_PLATFORMS="cpu", **extra_env)
    code = PROBE.format(imports=imports)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    lines = out.stdout.strip().splitlines()
    return lines[-1] if lines else out.stderr[-500:]


def test_a_program_takes_its_constants_as_arguments_by_default():
    """in either import order: most scripts import jax first"""
    assert _lowered("import jno\nimport jax") == "HOISTED"
    assert _lowered("import jax\nimport jno") == "HOISTED"


def test_the_opt_out_embeds_them():
    assert _lowered("import jno\nimport jax", JNO_HOIST_CONSTANTS="0") == "EMBEDDED"
    assert _lowered("import jax\nimport jno", JNO_HOIST_CONSTANTS="0") == "EMBEDDED"


def test_a_setting_for_jax_itself_is_kept():
    assert _lowered("import jno\nimport jax", JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS="0") == "EMBEDDED"
    assert _lowered("import jax\nimport jno", JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS="0") == "EMBEDDED"
