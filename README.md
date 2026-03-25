<p align="center">
  <img src="assets/logo.png" alt="jNO logo" width="500"/>
</p>

<p align="center">
    <a href="https://fhg-iisb.github.io/jNO_docs/">
        <img src="https://img.shields.io/badge/docs-GitHub%20Pages-0aa?style=for-the-badge" alt="Docs"/>
    </a>
    <a href="https://fhg-iisb.github.io/jNO_docs/Tutorials/">
        <img src="https://img.shields.io/badge/tutorials-step_by_step-0b8f7a?style=for-the-badge" alt="Tutorials"/>
    </a>
    <a href="https://github.com/FhG-IISB/jno/actions/workflows/python-package.yml">
        <img src="https://img.shields.io/github/actions/workflow/status/FhG-IISB/jno/python-package.yml?branch=main&style=for-the-badge&label=tests" alt="Tests"/>
    </a>
    <a href="LICENSE">
        <img src="https://img.shields.io/badge/license-MIT-2ea44f?style=for-the-badge" alt="License"/>
    </a>
    <a href="CITATION.cff">
        <img src="https://img.shields.io/badge/cite-CITATION.cff-6b5b95?style=for-the-badge" alt="Citation"/>
    </a>
    <img src="https://img.shields.io/badge/docker-image%20available-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker image available"/>
</p>


# Install

Quick install from PyPI:

```bash
pip install jax-neural-operators
```

For local development (recommended on Linux `aarch64` when `gmsh` wheels are unavailable on PyPI), use `micromamba`:

```bash
micromamba create -n jno python=3.12 pip -y
micromamba activate jno
micromamba install -n jno -c conda-forge gmsh python-gmsh -y
pip install -e .
```

### Foundation Models

These models are maintained as separate repositories so they can also be used independently.
If installed, you can access them via:

```python
jno.nn.<model_name>
```

to use a foundation model.

<p>
    <a href="https://github.com/FhG-IISB/jax_poseidon">
        <img src="https://img.shields.io/badge/jax_poseidon-1f6feb?style=for-the-badge" alt="jax_poseidon"/>
    </a>
    <a href="https://github.com/FhG-IISB/jax_walrus">
        <img src="https://img.shields.io/badge/jax_walrus-1f6feb?style=for-the-badge" alt="jax_walrus"/>
    </a>
    <a href="https://github.com/FhG-IISB/jax_pdeformer2">
        <img src="https://img.shields.io/badge/jax_pdeformer2-1f6feb?style=for-the-badge" alt="jax_pdeformer2"/>
    </a>
    <a href="https://github.com/armbrusl/jax_morph">
        <img src="https://img.shields.io/badge/jax_morph-1f6feb?style=for-the-badge" alt="jax_morph"/>
    </a>
    <a href="https://github.com/armbrusl/jax_mpp">
        <img src="https://img.shields.io/badge/jax_mpp-1f6feb?style=for-the-badge" alt="jax_mpp"/>
    </a>
</p>



## Citation

If jNO is used we would appreciate to cite the following paper:

```text
@article{armbruster2026jNO,
  author  = {Armbruster, Leon, ....},
  title   = {{jNO}: A JAX Library for Neural Operator and Foundation Model Training},
  journal = {},
  year    = {},
}
```

