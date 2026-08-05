# Miscellaneous

Utilities that sit outside the core PINN / solver workflow.

- **[Weights & Biases](misc/wandb.md)** — experiment tracking and logging.
- **[Hyperparameter Tuning](Hyperparameter-Tuning.md)** — sweeps over `net.tune(...)` options.
- **[Save, Load and Configuration](Save-Load-and-Configuration.md)** — persist models and runs with
  `jno.save` / `jno.load`.
- **[Glossary](Glossary.md)** — terminology reference.

!!! info "Looking for trackers, custom functions, trainable parameters, or the parameter Jacobian?"
    Those are operations on traced expressions and live under **[Operations](operations.md)** —
    see [Trackers](operations.md#trackers-labels-debugging),
    [Custom functions](operations.md#custom-functions),
    [Trainable scalar parameters](operations.md#trainable-scalar-parameters), and the
    [Parameter Jacobian & NTK](operations.md#parameter-jacobian-the-neural-tangent-kernel).
