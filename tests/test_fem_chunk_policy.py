"""The element-chunk saturation floor must not be applied where there is nothing to saturate.

`cell_chunk` bounds the batched element intermediate by bytes, then raises the result to
`_CHUNK_MIN_CELLS` so an accelerator does not run dry -- a trade that knowingly overruns the memory
cap because the measured alternative is a ~2x slowdown (see the sweep in `fem_utils`).

On a CPU backend there is no such trade to make: `_CHUNK_FALLBACK_BYTES` is a memory-only budget for
exactly that case. Applying the floor anyway enforces a GPU heuristic against a machine that cannot
benefit from it, and it inverts while doing so -- the floor counts CELLS while the resource is
BYTES, so it binds hardest on high-DOF elements and vanishes on meshes smaller than the floor, which
is the opposite of where anyone would look for a memory problem.

Measured on 4,082 tets of a 48-DOF enriched element with trainable coordinates: an 8 MiB budget, a
floor that disabled chunking outright, and a 36.2 GB peak against 13.4 GB once the cap was honoured.
"""

import os

import pytest

from jno.utils.solver import fem_utils


@pytest.fixture
def cpu_backend(monkeypatch):
    monkeypatch.setattr(fem_utils, "_device_saturates", lambda: False)
    monkeypatch.setattr(fem_utils, "chunk_budget_bytes", lambda: 8 << 20)


@pytest.fixture
def gpu_backend(monkeypatch):
    monkeypatch.setattr(fem_utils, "_device_saturates", lambda: True)
    monkeypatch.setattr(fem_utils, "chunk_budget_bytes", lambda: 8 << 20)


# n_test = n_local = 48 is the enriched (cover) vector tet: 4 nodes x (1 value + 3 covers) x 3
# components. Its per-cell block is 48**3 * 8 = 864 KiB, so an 8 MiB budget buys 9 cells.
COVER = dict(n_test=48, n_local=48)
LAGRANGE = dict(n_test=12, n_local=12)  # plain P1 vector tet: 12**3 * 8 = 13.5 KiB per cell


def test_cpu_honours_the_byte_cap_on_a_high_dof_element(cpu_backend):
    """4,082 cells at 864 KiB each must be split, not vmapped whole."""
    chunk = fem_utils.cell_chunk(4082, **COVER)
    assert chunk is not None, "chunking was skipped; the whole mesh would be materialised at once"
    assert chunk <= 4082
    assert chunk * 48 * 48 * 48 * 8 <= 4 * (8 << 20), (
        f"chunk of {chunk} cells is {chunk * 48 ** 3 * 8 / 2**20:.0f} MiB against an 8 MiB budget"
    )


def test_gpu_keeps_the_saturation_floor(gpu_backend):
    """Where there ARE cores to feed, the floor still wins -- that trade is measured, not assumed."""
    assert fem_utils.cell_chunk(4082, **COVER) is None  # 4082 <= 8192, one chunk
    # `_balanced_chunk` may land just UNDER the floor -- it returns the smallest chunk giving the
    # same piece count (40820 in 5 pieces is 8164, not 8192) -- so assert the floor DOMINATES the
    # 9-cell byte budget rather than asserting the exact value.
    assert fem_utils.cell_chunk(40820, **COVER) > 4000


def test_low_dof_elements_are_unaffected_on_cpu(cpu_backend):
    """A P1 tet fits ~620 cells in the same budget, so small meshes still take the single vmap."""
    assert fem_utils.cell_chunk(400, **LAGRANGE) is None


def test_the_floor_used_to_invert_on_small_meshes(cpu_backend):
    """The regression itself: smaller mesh, same element, must not become LESS bounded."""
    small = fem_utils.cell_chunk(4082, **COVER)
    large = fem_utils.cell_chunk(40820, **COVER)
    assert small is not None and large is not None
    assert small <= 4082 and large <= 40820
    # Per-chunk footprint is what must stay bounded, not the cell count.
    assert small <= large * 2, "the small mesh must not get a wildly larger chunk than the large one"


def test_explicit_setting_still_overrides_everything(gpu_backend):
    """`jno.fem(chunk=...)` is an upper bound the policy may not raise, on any backend."""
    assert fem_utils.cell_chunk(4082, setting=128, **COVER) <= 128
    assert fem_utils.cell_chunk(4082, setting=0, **COVER) is None


def test_cpu_budget_scales_with_host_ram(monkeypatch):
    """The CPU budget must track the machine, not a flat constant.

    A fixed 8 MiB is not conservative, it is arbitrary: on a 62 GB host it buys 9 cells of a 48-DOF
    element, and the saturation floor was what hid that. Spending the same fraction of host RAM the
    device path spends of device RAM lands on ~111 cells there -- next to the 128 measured best."""
    monkeypatch.setattr(fem_utils, "_device_saturates", lambda: False)
    monkeypatch.setattr(fem_utils.jax, "local_devices", lambda: [])  # force the no-device branch
    budget = fem_utils.chunk_budget_bytes()
    assert budget >= fem_utils._CHUNK_FALLBACK_BYTES, "never below the floor the flat value provided"
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    assert budget == max(fem_utils._CHUNK_FALLBACK_BYTES, int(total * fem_utils._CHUNK_MEMORY_FRACTION))
