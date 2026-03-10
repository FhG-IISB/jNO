<p align="center">
  <img src="assets/logo.png" alt="jNO logo" width="200"/>
</p>

<p align="center">
    <a href="https://fhg-iisb.github.io/jNO_docs/">
        <img src="https://img.shields.io/badge/docs-GitHub%20Pages-0aa?style=for-the-badge" alt="Docs"/>
    </a>
    <a href="LICENSE">
        <img src="https://img.shields.io/badge/license-MIT-2ea44f?style=for-the-badge" alt="License"/>
    </a>
    <a href="CITATION.cff">
        <img src="https://img.shields.io/badge/cite-CITATION.cff-6b5b95?style=for-the-badge" alt="Citation"/>
    </a>
</p>

## jNO

JAX library for neural operator and foundation model training.

## Install

```bash
pip install jno
```

## Citation

If you use jNO in your work, please cite:

```text
Leon Armbruster. jNO: A JAX Library for Neural Operator and Foundation Model Training. 2026. Version 0.1.0. (arXiv preprint)
```

See [CITATION.cff](CITATION.cff) for structured metadata.


## Dependencies (thank you!)

This project stands on the shoulders of some fantastic open-source libraries:

- Backbone → [JAX](https://github.com/jax-ml/jax)
- Optimizers → [Optax](https://github.com/google-deepmind/optax) and/or [SOAP](https://github.com/haydn-jones/SOAP_JAX)
- Neural Network → [Equinox](https://github.com/patrick-kidger/equinox) and/or [Flax](https://github.com/google/flax)
- Mesh generation → [pygmsh](https://github.com/nschloe/pygmsh) + [meshio](https://github.com/nschloe/meshio)
- Hyperparameter search → [Nevergrad](https://github.com/facebookresearch/nevergrad)
- Signed serialisation → [pylotte](https://github.com/FhG-IISB/pylotte)
- Einsum notation → [einops](https://github.com/arogozhnikov/einops)
