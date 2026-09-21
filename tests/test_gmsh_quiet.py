"""gmsh's progress output stays off the terminal; its warnings do not.

`Domain._generate_mesh` meshes inside a pygmsh context that has already started gmsh with the
terminal on, so `emit` must silence it whether or not it opened the session itself.
"""

import gmsh
import pytest

import jno
from jno.geometry import emit


def test_meshing_prints_no_gmsh_progress(capfd):
    # A 3-D cut is the loudest case: 53 lines of `Info    :` before the fix.
    d = (jno.shape.box(0, 0, 0, 1, 1, 1, size=0.4) - jno.shape.sphere(0.5, 0.5, 0.5, 0.2)).domain()
    d.variable("interior")
    out, err = capfd.readouterr()
    assert "Info    :" not in out + err


def test_a_gmsh_warning_during_a_build_reaches_the_jno_log(monkeypatch):
    # Silencing the terminal must not silence the warnings: a failed surface recovery or an
    # inverted element is something the user needs to see. No jNO shape triggers a gmsh warning
    # reliably, so one is planted INSIDE a real build, where the session is open.
    seen = []

    class Log:
        def warning(self, msg):
            seen.append(msg)

    monkeypatch.setattr("jno.utils.logger.get_logger", lambda *a, **k: Log())
    real = emit._emit_node

    def noisy(node, occ, split_full=False):
        gmsh.logger.write("a planted warning", level="warning")
        return real(node, occ, split_full)

    monkeypatch.setattr(emit, "_emit_node", noisy)
    mesh, dim, _ = emit.build(jno.shape.rect(0, 0, 1, 1, size=0.5))
    assert dim == 2 and len(mesh.points) > 4
    assert seen == ["gmsh: a planted warning"]


@pytest.mark.parametrize("terminal", [0, 1])
def test_a_callers_gmsh_session_keeps_its_terminal_setting(terminal):
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", terminal)
        emit.build(jno.shape.rect(0, 0, 1, 1, size=0.5))
        assert gmsh.option.getNumber("General.Terminal") == terminal
    finally:
        gmsh.finalize()
