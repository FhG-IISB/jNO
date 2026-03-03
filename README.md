# jNO
JAX-native building blocks for training neural operators.

> **Warning:** This is a research-level repository. It may contain bugs and is subject to continuous change without notice.

## [Install](https://docs.astral.sh/uv/getting-started/installation/)
`uv` installs and manages environments in your user directory, so you can typically run everything locally **without sudo**.
Local execution policies can be overwritten in windows as follows

```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
this will allow you to activate the environment.

```bash
uv sync --extra cuda
```

## Dependencies (thank you!)

This project stands on the shoulders of some fantastic open-source JAX ecosystem libraries:

- Backbone -> [JAX](https://github.com/jax-ml/jax)
- Optimizers -> [Optax](https://github.com/google-deepmind/optax) and/or [SOAP](https://github.com/haydn-jones/SOAP_JAX)
- Neural Network -> [Equinox](https://github.com/patrick-kidger/equinox) and/or [Flax](https://github.com/google/flax)
