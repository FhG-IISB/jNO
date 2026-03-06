# Logging and Callbacks

jNO provides built-in logging, debugging tools, and a callback system for monitoring and customizing the training process.

## Logger

### Initialization

Initialize the global logger at the start of your script. All jNO classes will log to this instance:

```python
import jno

dire = "./runs/my_experiment"
log = jno.logger(dire)
```

This creates:
- The output directory (if it doesn't exist)
- A log file at `{dire}/log.txt`
- Console output for real-time monitoring

### Custom Log Messages

```python
log("Starting training")
log("Custom message: experiment config loaded")
```

### Logger Configuration

The `Logger` class accepts:

```python
from jno.utils.logger import Logger

logger = Logger(
    path="./runs/experiment",   # Output directory
    log_print=(True, True),     # (log_to_file, print_to_console)
    name="log.txt"              # Log file name
)
```

---

## Training Progress Output

During training, jNO prints progress at regular intervals:

```
Epoch    100/10000 | C0: 1.2345e-02 | C1: 5.6789e-03
Epoch    200/10000 | C0: 8.1234e-03 | C1: 3.4567e-03
```

Where:
- `C0`, `C1`, ... are individual constraint losses
- `T0`, `T1`, ... are tracker values (if any)

### With Trackers

```python
val = jnn.tracker(jnn.mean(u - u_exact), interval=100)
```

When trackers are used, their values appear in the progress output:

```
Epoch    100/10000 | C0: 1.2345e-02 | T0: -2.3456e-04
```

---

## Debugging

### Shape and Value Inspection

Use the `.debug` attribute on constraint expressions to inspect intermediate values during training:

```python
# Available debug options
pde.debug._shape = True   # Print tensor shapes
pde.debug._val = True     # Print values
pde.debug._min = True     # Print minimum values
pde.debug._max = True     # Print maximum values
pde.debug._mean = True    # Print mean values
```

> **Warning:** Debugging with `jax.debug.print` is extremely expensive and should only be enabled temporarily for diagnosing issues.

### Model Summary

By default, wrapped models print a summary when first constructed. To suppress:

```python
net = jnn.nn.mlp(hidden_dims=64, num_layers=2)
net.dont_show()
```

---

## Callbacks

jNO supports a callback system for custom actions during training.

### Base Callback Class

```python
from jno.utils.callbacks import Callback

class MyCallback(Callback):
    def on_epoch_end(self, state, *args, **kwargs):
        """Called at the end of each epoch."""
        pass

    def on_training_end(self, state, *args, **kwargs):
        """Called at the end of training."""
        pass
```

The `state` argument is the `core` solver instance, giving access to parameters, losses, and other training state.

### Example: Early Stopping

```python
class EarlyStopping(Callback):
    def __init__(self, patience=100, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0

    def on_epoch_end(self, state, *args, **kwargs):
        current_loss = state.training_logs[-1]['total_loss']
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
```

---

## Training Statistics and Plotting

Each `solve()` call returns a statistics object:

```python
stats = crux.solve(10_000, optax.adam(1), lr_schedule)

# Plot training history
stats.plot("training_history.png")
```

The plot shows:
- Individual constraint losses over epochs
- Total loss
- Learning rate schedule
- Tracker values (if any)

---

## Computation Graph Visualization

Inspect the symbolic computation graph for any constraint:

```python
crux.visualize_trace(pde).save("trace_pde.dot")
crux.visualize_trace(bc).save("trace_bc.dot")
```

The generated `.dot` files can be visualized at [edotor.net](https://edotor.net/) or with Graphviz.

---

## Lox Integration

jNO uses [lox](https://github.com/huterguier/lox) for structured logging of training metrics. During each epoch, the following are logged:

- `epoch` — current epoch number
- `learning_rate` — current learning rate
- `losses` — individual constraint losses
- `weights` — constraint weights
- `total_loss` — sum of all losses
- `track_stats` — tracker values

---

## See Also

- [Training and Solving](Training-and-Solving.md) — the training loop and solve configuration
- [FAQ](FAQ.md) — troubleshooting common issues
