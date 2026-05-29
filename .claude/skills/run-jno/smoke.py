"""
jNO smoke driver — exercises the full solve pipeline end-to-end.

Usage:
    pixi run python .claude/skills/run-jno/smoke.py

Exit 0 = all checks passed.  Non-zero = something is broken.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import tempfile, pathlib
import jax
import optax
import foundax
import jno

print(f"[1/6] imports OK — jno {jno.__version__}")

# --- Domain: 1-D Poisson -u'' = sin(πx), u(0)=u(1)=0 ---
π = jno.np.pi
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
x, _ = domain.variable("interior")
xb, _ = domain.variable("boundary")
print("[2/6] domain OK")

# --- Network + model controls ---
net = jno.nn.wrap(
    foundax.mlp(in_features=1, hidden_dims=32, num_layers=2, key=jax.random.PRNGKey(0))
).optimizer(optax.adam(1e-3))
net.freeze()
net.reset()   # clear; now ready for normal training
net.optimizer(optax.adam(1e-3))
print("[3/6] model controls OK")

# --- Constraints + core ---
u = net(x)
pde = -u.d2(x, scheme="finite_difference") - jno.np.sin(π * x)
bc = net(xb)
crux = jno.core([pde.mse, bc.mse], domain)
print("[4/6] core compiled OK")

# --- Solve (10 epochs — just validates the pipeline) ---
history = crux.solve(10)
log = history.training_logs[-1]
total_loss = float(log["total_loss"][-1])
print(f"[5/6] solve OK — final loss {total_loss:.4e}")

# --- Eval + save/load ---
(u_pred,) = crux.eval([u])
assert u_pred.shape[0] > 0, f"eval returned empty array, shape={u_pred.shape}"

with tempfile.TemporaryDirectory() as tmp:
    p = pathlib.Path(tmp) / "model.pkl"
    jno.save(crux, str(p))
    crux2 = jno.load(str(p))
    (u_pred2,) = crux2.eval([u])
    assert u_pred2.shape == u_pred.shape, "save/load shape mismatch"

print("[6/6] eval + save/load OK")
print("\n✓ smoke test passed")
