# Save, Load & Configuration

## Save & Load

jNO uses [cloudpickle](https://github.com/cloudpipe/cloudpickle) to serialise the full solver state into a single file. The saved file contains:

- Model weights (all `jno.core.models`)
- Optimiser states
- Training logs
- The domain and its mesh
- The symbolic expression tree (constraints, trackers)
- Checkpoints from all previous `solve()` calls
- The RNG state

### Saving

```python
# Instance method
crux.save("runs/crux.pkl")

# Module-level function
import jno
jno.save(crux, "runs/crux.pkl")
```

### Loading

```python
# Module-level function
crux = jno.load("runs/crux.pkl")

# Module-level function
crux = jno.load("runs/crux.pkl")
```

### Continuing Training After Load

After loading, all Python variable references (e.g., `u_net`, `v_net`) no longer point to the models inside the loaded solver. Use `set_optimizer` to reassign optimisers to all models at once:

```python
crux = jno.load("runs/crux.pkl")
crux.set_optimizer(optax.adam, scale=lrs(1e-5))
crux.solve(1000).plot("continued.png")
```

### Checkpoints During Training — `restore_checkpoint`

`crux.save()` above writes the whole solver in one file at a moment you choose. A **checkpoint** is
the other thing: written periodically *during* `solve()` by a callback, holding weights, optimiser
state and the RNG, so a long run survives an interruption.

```python
cb = jno.callbacks.checkpoint(
    directory="runs/ckpt",
    save_interval_epochs=500,
    max_to_keep=3,
    best_fn=lambda m: m["total_loss"],   # keep the best one regardless of age
)
crux.solve(20_000, callbacks=[cb])
```

Restoring is two lines, and the second is not optional:

```python
crux.restore_checkpoint("runs/ckpt")   # latest; or step=2000 for a specific one
crux.solve(1)                          # the ARRAYS are restored here, not above
```

!!! warning "The restore is deferred to the next `solve()`"
    `restore_checkpoint` reads the metadata immediately — the epoch counter moves straight away — but
    schedules the weight restore for the next `solve()` call. Orbax needs the live Equinox/Optax tree
    as a target, and that only exists inside `solve()` after the partition and optimiser setup. So a
    `crux.eval(...)` between the two lines still sees the **old** weights.

    The epoch counter is set to the **checkpoint's** step, not to where you were. Restoring step 500
    of a 2000-epoch run and calling `solve(1)` leaves you at epoch 501.

!!! danger "Restore into the solver that wrote it"
    Restoring into a **freshly constructed** `jno.core` — even one built by the same code with the
    same architecture — raises:

    ```
    ValueError: User-provided restore item and on-disk value metadata tree structures
    do not match: opt_states.1: - Source: MISSING
    ```

    The optimiser-state tree is keyed by layer id, and a new solver assigns new ones. Use
    `crux.save()` / `jno.load()` to move a run between processes; `restore_checkpoint` is for
    rewinding the solver that produced the checkpoint.

`jno.core(..., resume_from="runs/ckpt")` is the same mechanism as a constructor argument, consumed on
the first `solve()`.

### Evaluation After Load

```python
crux = jno.load("runs/crux.pkl")

# Re-evaluate any constraint expression on the training domain
pred = crux.eval(u)

# Evaluate on a different domain
test_domain = jno.Shape.rect(0, 0, 1, 1, size=0.01).domain()
pred_fine = crux.eval(u, domain=test_domain)
```

---

## Encrypted (RSA-Signed) Save / Load

jNO integrates [pylotte](https://github.com/FhG-IISB/pylotte) for RSA-signed serialisation. This ensures the integrity of saved models: a `.sig` file is created alongside the `.pkl` file, and loading verifies the signature.

### Generating Keys

Use OpenSSL (or any RSA key generation tool):

```bash
# Generate a 4096-bit RSA private key
openssl genrsa -out ~/.jno/private.pem 4096

# Extract the public key
openssl rsa -in ~/.jno/private.pem -pubout -out ~/.jno/public.pem
```

### Configuring Keys

Add to `.jno.toml` (see [Configuration](#configuration) below):

```toml
[rsa]
public_key  = "~/.jno/public.pem"
private_key = "~/.jno/private.pem"
```

When keys are configured, `save` and `load` use them automatically.

### Explicit Key Paths

```python
jno.save(crux, "runs/crux.pkl",
         public_key_path="~/.jno/public.pem",
         private_key_path="~/.jno/private.pem")

crux = jno.load("runs/crux.pkl",
                public_key_path="~/.jno/public.pem",
                signature_path="runs/crux.sig")
```

---

## Configuration

jNO looks for a TOML configuration file in two locations (first match wins):

1. `.jno.toml` in the **current working directory** (project-level)
2. `~/.jno/config.toml` (user-level)

All fields are optional.

### Example `.jno.toml`

```toml
[jno]
seed = 42              # global RNG seed (used by all jno.core instances)

[runs]
base_dir = "./runs"    # base directory for jno.setup() run directories

[rsa]
public_key  = "~/.jno/public.pem"    # RSA public key for signed save/load
private_key = "~/.jno/private.pem"   # RSA private key
```

### Configuration API

```python
import jno

# Force reload the config from disk
jno.load_config(force=True)

# Get the current (cached) config dict
cfg = jno.get_config()

# Get the path of the loaded config file (None if not found)
path = jno.get_config_path()

# Individual accessors
runs_dir = jno.get_runs_base_dir()        # default: "./runs"
seed     = jno.get_seed()                 # None if not set
pub_key  = jno.get_rsa_public_key()       # None if not configured
priv_key = jno.get_rsa_private_key()      # None if not configured
```

### Project Setup Helper

```python
# Create the run directory and initialise logging in one call
dire = jno.setup(__file__)

# Override the subdirectory name
dire = jno.setup(__file__, name="experiment_v3")
```

`setup()` derives the subdirectory name from the calling script's filename stem (e.g., `heat_equation.py` → `./runs/heat_equation/`). A global RNG seed set in the config is automatically picked up by all `jno.core` instances created after `setup()`.

---

## Logging

jNO includes a structured logger. By default `jno.setup()` initialises it:

```python
dire = jno.setup(__file__)

# Write custom log messages
log = jno.logger(dire)
log("Starting experiment")
log.info("Epoch 1000 reached")
log.warning("Large loss spike detected")
```

Log files are written to the run directory.

---

## IREE Ahead-of-Time Compilation

For deployment scenarios, jNO supports exporting compiled models via [IREE](https://iree.dev/):

Install IREE support:
```bash
pip install jax-numerical-operators[iree]
```

```python
from jno import iree as IREEModel

# Compile and export
compiled = IREEModel(crux, ...)
compiled.save("model.vmfb")

# Load and run
loaded = IREEModel.load("model.vmfb")
result = loaded(inputs)
```
