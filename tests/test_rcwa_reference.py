"""jNO's RCWA against independent libraries on the same truncated problem.

``tests/reference/rcwa_reference.json`` holds T and R that torcwa (Laurent's rule, ``order=[M, M]``) and fmmax
(``Formulation.FFT``, parallelogramic truncation) computed for the cases in ``tests/reference/cases.py``; it is
regenerated in the separate ``rcwa-ref`` environment (``tests/reference/README.md``), and this test needs neither
library. With ``formulation="FFT"`` and the parallelogramic truncation jNO solves exactly that problem, so the
efficiencies must agree to the reference's own roundoff: measured 4e-11 to 8e-10 against torcwa (whose eigensolve
sets that floor), 7e-10 against fmmax.

fmmax is used only at normal incidence: at oblique incidence, driven as here (and as jNO's main drives it), its
T + R exceeds 1 on these lossless cases by up to 8 % and T is off by up to 0.10 (git-bug a20b207), while torcwa
and jNO conserve energy to roundoff.
"""

import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference")
sys.path.insert(0, HERE)
from cases import CASES, grid_of  # noqa: E402

with open(os.path.join(HERE, "rcwa_reference.json")) as fh:
    REFERENCE = json.load(fh)["results"]


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _jno_tr(case, m):
    from jno._rcwa import basis
    from jno._rcwa.stack import StateStack

    k0 = 2 * np.pi / case["wavelength"]
    st = StateStack(
        case["wavelength"],
        case["period"],
        case["eps_top"],
        case["eps_bottom"],
        [(d, jnp.asarray(grid_of(spec))) for d, spec in case["layers"]],
        orders=(2 * m + 1) ** 2,
        k_in=(case["k_in"][0] * k0, case["k_in"][1] * k0),
        formulation="FFT",
        dtype=jnp.complex128,
        truncation=basis.Truncation.PARALLELOGRAMIC,
    )
    assert st.num_terms == (2 * m + 1) ** 2
    res = st.solve(polarization=case["polarization"], validate=False)
    return float(res.T), float(res.R)


PAIRS = [(name, m) for name, case in CASES.items() for m in case["truncations"]]


@pytest.mark.parametrize("name,m", PAIRS, ids=[f"{n}-M{m}" for n, m in PAIRS])
def test_jno_equals_torcwa_on_the_same_truncated_problem(name, m):
    t, r = _jno_tr(CASES[name], m)
    ref = REFERENCE[name][str(m)]["torcwa"]
    assert t == pytest.approx(ref["T"], abs=2e-9)
    assert r == pytest.approx(ref["R"], abs=2e-9)


@pytest.mark.parametrize("name,m", [(n, m) for n, m in PAIRS if CASES[n]["k_in"] == (0.0, 0.0)])
def test_jno_equals_fmmax_at_normal_incidence(name, m):
    t, r = _jno_tr(CASES[name], m)
    ref = REFERENCE[name][str(m)]["fmmax"]
    assert t == pytest.approx(ref["T"], abs=2e-9)
    assert r == pytest.approx(ref["R"], abs=2e-9)
