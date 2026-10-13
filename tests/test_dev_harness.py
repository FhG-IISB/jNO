"""The developer profiling harness in dev/ (not shipped): the trace-to-stage join and the ncu CSV parse, on fixtures,
without a GPU."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
pytest.importorskip("matplotlib")
from dev import kernels, prof  # noqa: E402


def test_stage_of_forward_transpose_and_nested_custom_linear_solve():
    assert prof._stage_of("jit(f)/jvp(jno/assemble)/dot_general") == ("jno/assemble", False)
    assert prof._stage_of("jit(f)/transpose(jvp(jno/assemble))/dot_general") == ("jno/assemble", True)
    assert prof._stage_of("jit(L)/jvp()/jno/solve/ffi_call") == ("jno/solve", False)
    assert prof._stage_of("jit(L)/transpose(jvp())/jno/solve/ffi_call") == ("jno/solve", True)
    assert prof._stage_of("jit(L)/jvp()/mul") == ("other", False)


def test_hlo_meta_maps_fusions_to_their_scope():
    hlo = """
  %loop_gather_fusion = f64[512,1,1]{2,1,0} fusion(%b.1), kind=kLoop, calls=%fused_gather, metadata={op_name="jit(f)/jvp(jno/solve)/gather" stack_frame_id=3}
  ROOT %gemm_fusion_dot_general.7 = f64[512,512]{1,0} fusion(%a.1), kind=kCustom, metadata={op_name="jit(f)/transpose(jvp(jno/assemble))/dot_general" stack_frame_id=4}, backend_config={}
"""
    m = prof._hlo_meta(hlo)
    assert m["loop_gather_fusion"] == "jit(f)/jvp(jno/solve)/gather"
    assert m["gemm_fusion_dot_general.7"] == "jit(f)/transpose(jvp(jno/assemble))/dot_general"


def test_nvtx_regex_escapes_the_scope_separator():
    # ncu splits the NVTX stack on '/', so the '/' inside jno/solve must match any character; push/pop needs the trailing '/'
    assert kernels.nvtx_regex("jno/solve") == "regex:TSL@Thunk.*jno.solve.*/"


def test_parse_csv_wide_layout_units_and_derived_fields():
    hdr = [
        '"ID"',
        '"Kernel Name"',
        '"Grid Size"',
        '"Block Size"',
        f'"{kernels.NVTX_COL}"',
        '"gpu__time_duration.sum"',
        '"sm__sass_thread_inst_executed_op_dfma_pred_on.sum"',
        '"sm__sass_thread_inst_executed_op_dadd_pred_on.sum"',
        '"dram__bytes_read.sum"',
        '"dram__bytes_write.sum"',
        '"lts__t_sector_hit_rate.pct"',
    ]
    units = ['""', '""', '""', '""', '""', '"ms"', '"inst"', '"inst"', '"Mbyte"', '"Kbyte"', '"%"']
    row = [
        '"3"',
        '"void cudss::fwd_ker<double2>(int)"',
        '"(404, 1, 1)"',
        '"(128, 1, 1)"',
        '"1 ""TSL:Thunk:#name=jit(L)/jvp()/jno/solve/ffi_call,hlo_op=ffi_call.4#:none"" "',
        '"0.5"',
        '"1,000"',
        '"500"',
        '"2"',
        '"500"',
        '"75.5"',
    ]
    text = "\n".join(",".join(r) for r in (hdr, units, row)) + "\n"
    (rec,) = kernels.parse_csv(text)
    assert rec["id"] == 3 and rec["hlo_op"] == "ffi_call.4" and rec["scope"] == "jit(L)/jvp()/jno/solve/ffi_call"
    assert rec["time_ns"] == pytest.approx(0.5e6)
    assert rec["fp64_flop"] == 2 * 1000 + 500
    assert rec["dram_B"] == pytest.approx(2e6 + 500e3)
    assert rec["l2_hit_pct"] == pytest.approx(75.5)
    assert rec["intensity"] == pytest.approx(2500 / 2.5e6)
    assert rec["fp64_flop_per_s"] == pytest.approx(2500 / 0.5e-3)


def test_intrinsic_bounds_from_sizes_and_backend_record():
    r = prof.Result(
        "x",
        {"sizes": dict(cells=1000, nodes=250, nnz=20000, n=2000, nrhs=3, complex=True)},
        {},
        [],
        [],
        [{"backend": "cudss_ffi", "lu_nnz": 100000}],
    )
    pk = {"fp64_flop_s": 1e9, "dram_B_s": 1e9}
    b = r.intrinsic_bounds(pk)
    mesh_B, trip_B = 1000 * 16 + 250 * 24, 20000 * (16 + 8)
    assert b["jno/assemble"] == pytest.approx(2 * (mesh_B + trip_B) / 1e9 * 1e3)
    assert b["jno/factor"] == pytest.approx(100000**2 / 2000 / 1e9 * 4 * 1e3)
    assert b["jno/solve"] == pytest.approx(2 * 2 * 3 * 100000 * 16 / 1e9 * 1e3)
    assert prof.Result("y", {"sizes": {}}, {}, [], [], []).intrinsic_bounds(pk) == {}


def test_cpu_samples_and_attribution(tmp_path, monkeypatch):
    from dev import cpu

    script = (
        "    100003      cpu_core/fp_arith_inst_retired.scalar_double/:      7975fda9eb85 multiply_reduce_fusion.3\n"
        "    100003      cpu_core/fp_arith_inst_retired.256b_packed_double/:      7975fda9eb90 multiply_reduce_fusion.3\n"
        "        63                             cpu_core/cycles/:  ffffffffa9283d92 __perf_event_task_sched_in\n"
        "      1000                             cpu_core/cycles/:  00007975fda9eb85 multiply_reduce_fusion.3\n"
        "      2000                             cpu_core/cycles/:  00007975fdaa0000 dgemm_\n"
        "garbage line\n"
    )
    monkeypatch.setattr(
        cpu.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": script, "returncode": 0, "stderr": ""})()
    )
    agg = cpu.samples("perf.data")
    assert agg[("fp_arith_inst_retired.scalar_double", "multiply_reduce_fusion.3")] == 100003
    assert agg[("cycles", "dgemm_")] == 2000
    meta = {"multiply_reduce_fusion.3": "jit(f)/transpose(jvp(jno/assemble))/mul"}
    by, syms = cpu.attribute(agg, meta, ["cycles", "fp64_scalar", "fp64_256b"])
    assert by["jno/assemble"]["cycles"] == 1000 and by["jno/assemble"]["fp64_256b"] == 100003
    assert by["library"]["cycles"] == 2000 + 63
    assert syms["jno/assemble"][("multiply_reduce_fusion.3", "fp64_scalar")] == 100003


def test_verdicts_follow_the_rungs():
    K = prof.Kernel
    mk = lambda stage, ms, n, adj=False: K("op", "k", f"jit(f)/jvp(jno/{stage[4:]})/x", stage, adj, ms * 1e6 / n, n)
    r = prof.Result(
        "x",
        {"sizes": dict(cells=1000, nodes=250, nnz=20000, n=2000, nrhs=1, complex=False)},
        {},
        [mk("jno/factor", 100, 10), mk("jno/assemble", 50, 10000), mk("jno/solve", 20, 10), mk("jno/readout", 0.5, 5)],
        [],
        [{"backend": "cudss_ffi", "lu_nnz": 100000}],
    )
    v = r.verdicts({"jno/solve": (18.0, None, 10), "jno/factor": (30.0, None, 10)})
    assert v["jno/assemble"].startswith("launch-bound")  # 10,000 launches x 4 us = 40 ms of 50
    assert v["jno/factor"].startswith("x3.3 above the roof")  # 100 / 30
    assert "reformulate" in v["jno/solve"] or "at the roof" in v["jno/solve"]  # 20 / 18 < 1.5
    assert v["jno/readout"].startswith("negligible")
