# Installation

## Prerequisites

- Python 3.11 – 3.12.4
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended package manager)

## Install with uv

`uv` installs and manages environments in your user directory, so you can typically run everything locally **without sudo**.

```bash
uv sync
```

### CUDA Support

To enable GPU acceleration via CUDA:

```bash
uv sync --extra cuda
```

### Development Dependencies

To install development tools (pytest, black, flake8, mypy):

```bash
uv sync --extra dev
```

## Windows Notes

On Windows, local execution policies may block activation of virtual environments. Run the following in PowerShell to allow it:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Linux Notes

If you encounter errors related to OpenGL, install the required system library:

```bash
sudo apt-get install libglu1
```

## Verifying the Installation

After installation, verify that jNO imports correctly:

```python
import jno
import jno.numpy as jnn
print(jno.__version__)
```

## Project Structure

```
jNO/
├── jno/                    # Main package
│   ├── architectures/      # Neural operator architectures (FNO, U-Net, DeepONet, etc.)
│   ├── resampling/         # Adaptive resampling strategies
│   ├── utils/              # Logging, LoRA, callbacks, statistics
│   ├── core.py             # Core solver (training loop, optimization)
│   ├── core_utilities.py   # Plotting, error metrics, visualization
│   ├── domain.py           # Domain definition and meshing
│   ├── numpy.py            # NumPy-like API (math, differential operators)
│   ├── trace.py            # Symbolic computation graph tracing
│   ├── trace_evaluator.py  # Trace compilation and evaluation
│   └── tuner.py            # Architecture and hyperparameter search
├── examples/               # Example scripts
├── assets/                 # Logo and static assets
├── pyproject.toml          # Project metadata and dependencies
└── README.md
```
