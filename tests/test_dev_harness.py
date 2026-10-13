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
