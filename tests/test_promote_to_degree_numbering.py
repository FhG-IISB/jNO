"""``_promote_to_degree`` numbers its nodes exactly as the loop it replaced: the vertices first, keeping their ids, then each
new node at its first appearance, cell by cell. Oracle: that loop, written out here, with every kind of key and cell.
The numbering matters beyond the mesh itself: saved fields, tests and a remesh's transfer all read these ids."""

import numpy as np
import pytest

from jno.utils.solver.fem_utils import _promote_to_degree, _row_groups


def _loop(points, cells_p1, ref_pts, cell_type=None, *, entity_keys=False):
    points, cells_p1, ref_pts = np.asarray(points, float), np.asarray(cells_p1), np.asarray(ref_pts, float)
    ncell, ncorner = cells_p1.shape
    ndof = ref_pts.shape[0]
    if cell_type is not None:
        from jno.utils.solver.fem_lagrange import _lagrange_basix, basix_cell

        weights = np.asarray(_lagrange_basix(basix_cell(cell_type)[0], 1).tabulate(0, ref_pts))[0, :, :, 0]
    else:
        weights = np.empty((ndof, ncorner))
        weights[:, 0] = 1.0 - ref_pts.sum(axis=1)
        weights[:, 1:] = ref_pts
    phys = np.einsum("dc,ncg->ndg", weights, points[cells_p1])
    tol = 1e-7 * float(np.max(points.max(axis=0) - points.min(axis=0)))
    wtol, wround = 1e-9, 1e-6

    def key(c, d):
        if not entity_keys:
            return tuple(np.round(phys[c, d] / tol).astype(np.int64))
        w = weights[d]
        nz = np.flatnonzero(np.abs(w) > wtol)
        vids = cells_p1[c][nz]
        order = np.argsort(vids, kind="stable")
        return (tuple(int(v) for v in vids[order]), tuple(np.round(w[nz][order] / wround).astype(np.int64)))

    seen, out = {}, []
    for vid in range(points.shape[0]):
        seed = ((vid,), (int(round(1.0 / wround)),)) if entity_keys else tuple(np.round(points[vid] / tol).astype(np.int64))
        seen[seed] = vid
        out.append(points[vid])
    cells = np.zeros((ncell, ndof), dtype=np.int64)
    nid = points.shape[0]
    for c in range(ncell):
        for d in range(ndof):
            k = key(c, d)
            gid = seen.get(k)
            if gid is None:
                gid, seen[k] = nid, nid
                out.append(phys[c, d])
                nid += 1
            cells[c, d] = gid
    return np.asarray(out), cells


def _lattice(dim, n=4, shuffle=True, seed=0):
    rng = np.random.default_rng(seed)
    axes = [np.linspace(0.0, 1.0, n + 1)] * dim
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, dim)
    ids = np.arange(pts.shape[0]).reshape(*(n + 1,) * dim)
    if dim == 2:
        q = np.stack([ids[:-1, :-1], ids[1:, :-1], ids[1:, 1:], ids[:-1, 1:]], -1).reshape(-1, 4)
        tri = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
        return pts, (tri[rng.permutation(len(tri))] if shuffle else tri)
    h = np.stack(
        [
            ids[:-1, :-1, :-1],
            ids[1:, :-1, :-1],
            ids[1:, 1:, :-1],
            ids[:-1, 1:, :-1],
            ids[:-1, :-1, 1:],
            ids[1:, :-1, 1:],
            ids[1:, 1:, 1:],
            ids[:-1, 1:, 1:],
        ],
        -1,
    ).reshape(-1, 8)
    tets = np.concatenate(
        [h[:, list(t)] for t in ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))]
    )
    return pts, (tets[rng.permutation(len(tets))] if shuffle else tets)


def _ref(cell, k):
    import basix

    from jno.utils.solver.fem_lagrange import _lagrange_basix

    kind = {"triangle": basix.CellType.triangle, "tetrahedron": basix.CellType.tetrahedron}[cell]
    return np.asarray(_lagrange_basix(kind, k).points)


@pytest.mark.parametrize("entity_keys", [False, True], ids=["coordinates", "entities"])
@pytest.mark.parametrize(("dim", "cell"), [(2, "triangle"), (3, "tetrahedron")])
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_the_numbering_is_the_loops(dim, cell, k, entity_keys):
    pts, cells = _lattice(dim, n=3)
    ref = _ref(cell, k)
    got_pts, got_cells = _promote_to_degree(pts, cells, ref, entity_keys=entity_keys)
    want_pts, want_cells = _loop(pts, cells, ref, entity_keys=entity_keys)
    np.testing.assert_array_equal(got_cells, want_cells)
    np.testing.assert_array_equal(got_pts, want_pts)
    assert got_cells.dtype == want_cells.dtype


def test_two_bodies_on_one_interface_weld_by_coordinate_and_not_by_entity():
    """the two meanings of a coincident node: both as the loop had them, and they do differ"""
    pts, cells = _lattice(2, n=2, shuffle=False)
    other = pts + np.array([1.0, 0.0])
    both_pts = np.concatenate([pts, other])
    both_cells = np.concatenate([cells, cells + pts.shape[0]])
    ref = _ref("triangle", 2)
    for entity_keys in (False, True):
        got_pts, got_cells = _promote_to_degree(both_pts, both_cells, ref, entity_keys=entity_keys)
        want_pts, want_cells = _loop(both_pts, both_cells, ref, entity_keys=entity_keys)
        np.testing.assert_array_equal(got_cells, want_cells)
        np.testing.assert_array_equal(got_pts, want_pts)
    assert len(_promote_to_degree(both_pts, both_cells, ref, entity_keys=True)[0]) > len(
        _promote_to_degree(both_pts, both_cells, ref, entity_keys=False)[0]
    )


def test_a_quadrilateral_mesh():
    from jno.utils.solver.fem_lagrange import _lagrange_basix, basix_cell

    pts, _tri = _lattice(2, n=3, shuffle=False)
    ids = np.arange(pts.shape[0]).reshape(4, 4)
    quads = np.stack([ids[:-1, :-1], ids[1:, :-1], ids[:-1, 1:], ids[1:, 1:]], -1).reshape(-1, 4)  # basix vertex order
    ref = np.asarray(_lagrange_basix(basix_cell("quad")[0], 2).points)
    got_pts, got_cells = _promote_to_degree(pts, quads, ref, "quad")
    want_pts, want_cells = _loop(pts, quads, ref, "quad")
    np.testing.assert_array_equal(got_cells, want_cells)
    np.testing.assert_array_equal(got_pts, want_pts)


@pytest.mark.parametrize(
    "rows",
    [
        np.zeros((0, 3), np.int64),
        np.array([[7, 8, 9]], np.int64),
        np.tile(np.array([[1, 2]], np.int64), (5, 1)),
        np.arange(30, dtype=np.int64).reshape(-1, 3),
        np.random.default_rng(0).integers(-3, 3, (2000, 3)),
    ],
)
def test_row_groups_against_a_dict(rows):
    group, first = _row_groups(rows)
    seen = {}
    for i, row in enumerate(map(tuple, rows)):
        seen.setdefault(row, []).append(i)
    assert len(first) == len(seen)
    for g in range(len(first)):
        members = np.flatnonzero(group == g)
        assert seen[tuple(rows[members[0]])] == members.tolist()
        assert first[g] == members[0]


def test_a_hash_collision_is_not_a_merge(monkeypatch):
    """every row hashes alike (a multiplier of zero): only the fallback can tell them apart"""
    import jno.utils.solver.fem_utils as F

    monkeypatch.setattr(F, "_ROW_HASH_MULTIPLIER", np.uint64(0))
    rows = np.random.default_rng(1).integers(0, 3, (500, 3))
    group, first = _row_groups(rows)
    seen = {}
    for i, row in enumerate(map(tuple, rows)):
        seen.setdefault(row, []).append(i)
    assert len(first) == len(seen)
    for g in range(len(first)):
        assert seen[tuple(rows[np.flatnonzero(group == g)[0]])] == np.flatnonzero(group == g).tolist()
