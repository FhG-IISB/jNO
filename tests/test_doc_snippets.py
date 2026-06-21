"""Execute the self-contained code block on the hand-written FEM narrative page so its
front-page snippet can't silently rot. (The per-tutorial *full scripts* are already run by the
subprocess tutorial tests via the ``--8<--`` includes; the inline excerpts elsewhere are
fragments that reference setup vars and are intentionally not executed here.)
"""

import re
from pathlib import Path

import jax
import pytest

pytest.importorskip("feax", reason="feax required for the FEM doc snippet")
pytest.importorskip("shapely", reason="shapely required for the box domain")

DOCS = Path(__file__).parent.parent / "docs"


def _first_python_block(md_path: Path) -> str:
    m = re.search(r"```python\n(.*?)```", md_path.read_text(), re.DOTALL)
    assert m is not None, f"no ```python block found in {md_path}"
    return m.group(1)


def test_fem_md_intro_snippet_runs():
    """The fem.md front-page intro is a complete, self-contained jno.fem solve; it must execute.
    The headline API hands back ready-to-use arrays (fem.A dense, fem.b flat) for jnp.linalg.solve;
    this guards against drift in that intro and the flat-accessor contract."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)  # feax assembly is float64
    try:
        ns: dict = {}
        exec(compile(_first_python_block(DOCS / "fem.md"), "<fem.md intro>", "exec"), ns)
        u_h = ns["u_h"]
        assert u_h.ndim == 1 and u_h.shape[0] > 0  # solved a real linear system
    finally:
        jax.config.update("jax_enable_x64", prev)
