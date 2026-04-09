---
title: 'jNO: A JAX Library for Neural Operator and Foundation Model Training'
tags:
  - Python
  - JAX
  - neural operators
  - scientific machine learning
  - physics-informed machine learning
  - PDE foundation models
authors:
  - name: Leon Armbruster
    corresponding: true
    affiliation: 1
affiliations:
  - name: Fraunhofer IISB, Germany
    index: 1
date: 11 March 2026
bibliography: references.bib
---

# Summary

jNO is a JAX-native library for neural operators with unified support for both data-driven and physics-informed training. Its central design choice is a tracing system in which domains, model calls, residual terms, supervised losses, and diagnostics are written in one symbolic language and compiled into one optimization pipeline. The traced program is then JIT-compiled through JAX and XLA, allowing users to move between operator regression and PDE-constrained training without restructuring the surrounding code [@bradbury2018jax].

The library is designed for mesh-aware scientific machine learning workflows. Meshes can be generated through PyGmsh and Gmsh or loaded from external files, then exposed through a common domain interface for residual evaluation, supervision, and operator learning [@schlomer2018pygmsh; @geuzaine2009gmsh; @nschloe2024meshio]. jNO also supports multi-model compositions, fine-grained optimizer and parameter controls, LoRA-style adaptation, hyperparameter tuning, and persistence of trained experiments. A further objective of the project is to make neural-operator and PDE foundation-model workflows, which are often concentrated in PyTorch-first ecosystems, available in a JAX-native stack.

# Statement of need

Neural operators are increasingly used for surrogate modeling, inverse problems, and PDE-constrained learning [@li2021fno; @lu2021deeponet]. In practice, however, the surrounding software workflow is often fragmented across separate abstractions for geometry, loss construction, model definition, and training logic. This fragmentation becomes more pronounced when users want to combine supervised operator learning with physics-informed residuals or transfer a pretrained backbone into a new PDE setting.

jNO addresses this gap by treating the workflow as one traced program. Users define domains and variables, compose model and differential expressions, and solve everything through a single interface. This makes it possible to express hybrid objectives, multi-model programs, and mesh-based residuals without adding a separate orchestration layer for each training mode. The project therefore targets two connected use cases: unified neural-operator training that combines data supervision and physics-informed losses, and transfer or fine-tuning workflows for PDE foundation backbones translated to JAX.

# State of the field

DeepXDE remains an important reference for compact equation-centric PINN software design [@lu2021deepxde]. JAX-PI provides a JAX-native alternative focused specifically on PINN workflows [@predictive2024jaxpi]. NVIDIA PhysicsNeMo and NeuralPDE.jl provide complementary ecosystems for physics-based deep learning and scientific machine learning [@hennigh2021nvidia; @zubov2021neuralpde].

jNO differs from these tools by centering on a single traced language for neural operators and hybrid objectives in JAX. Rather than separating operator learning, physics constraints, and mesh-aware data handling into distinct APIs, it exposes them through the same symbolic layer. It also places stronger emphasis on JAX-native translation and adaptation of PDE foundation-model families, including Poseidon, Walrus, PDEformer2, MPP, and Morph [@fhgiisb2026jaxposeidon; @fhgiisb2026jaxwalrus; @fhgiisb2026jaxpdeformer2; @armbrusl2026jaxmpp; @armbrusl2026jaxmorph].

# Software design

The tracing layer in jNO is implemented as a symbolic DSL centered on `Placeholder` nodes. Arithmetic, slicing, reductions, and differential operators are overloaded to construct expression trees rather than execute eagerly. Many of these operations are thin symbolic wrappers over `jax.numpy`, so traced expressions remain close to ordinary JAX tensor semantics while deferring execution until compilation time. Objects such as variables, constants, models, Jacobians, and Hessians are therefore represented as nodes in one deferred graph.

Model invocations are first-class graph nodes. Training controls such as freezing, masking, optimizer attachment, LoRA activation, dtype selection, and per-group optimization settings are attached to model objects and remain part of the same symbolic programming flow as PDE and data terms. Before execution, the traced tree is normalized and optimized with common sub-expression elimination, reducing redundant work in both compile-time and runtime execution.

The `domain` layer acts as the mesh-aware bridge between geometry definitions and traced expressions. It supports mesh generation from built-in constructors through PyGmsh and Gmsh, loading external meshes through `meshio`, tagged point sets for interior and boundary subsets, batched contexts for neural-operator training, tensor tags for parametric inputs, and precomputed connectivity metadata for finite-difference evaluation [@schlomer2018pygmsh; @geuzaine2009gmsh; @nschloe2024meshio]. Adaptive resampling can update point sets during training without changing the symbolic equations.

All neural-operator architectures are exposed through a single model factory in `jno.numpy.nn`. Available model families include spectral methods, convolutional architectures, attention-based methods, branch-trunk decompositions, and point-based models. Custom Equinox and Flax modules can be wrapped into the same tracing and optimization pipeline [@kidger2021equinox; @heek2024flax]. In `core.py`, the solver compiles traced constraints into one JIT-compiled training program, supports multi-device execution, explicit sharding, buffer donation, gradient checkpointing, data offloading, fused inner steps, bounded profiling, and double-precision execution via `JAX_ENABLE_X64=1` [@bradbury2018jax]. Hyperparameter tuning is integrated through `ArchSpace`, with support for grid search and gradient-free optimization through Nevergrad [@facebook2018nevergrad].

The library also provides object-level persistence through `jno.save` and `jno.load` for solver states, domains, and exported IREE inference wrappers [@openxla2019iree; @cloudpipe2024cloudpickle]. Optional RSA-signed artifacts are supported through `pylotte` for verifiable sharing workflows [@alpamayo2026pylotte].

# Research impact statement

jNO is intended as a reusable JAX-native research software stack for neural-operator and PDE-constrained learning workflows that would otherwise be split across separate tooling. Its impact is reflected in three concrete ways. First, the library is packaged for installation through PyPI, which lowers the barrier for reuse in reproducible computational workflows. Second, the repository includes automated tests and executable examples that serve as reference implementations for operator-learning and physics-informed training scenarios. Third, the project acts as the common software base for related JAX translations of PDE foundation-model families, including Poseidon, Walrus, PDEformer2, MPP, and Morph [@fhgiisb2026jaxposeidon; @fhgiisb2026jaxwalrus; @fhgiisb2026jaxpdeformer2; @armbrusl2026jaxmpp; @armbrusl2026jaxmorph]. Together, these distribution, validation, and extension pathways provide credible near-term research significance by making advanced neural-operator workflows more accessible within the JAX ecosystem.

# Quality control

jNO uses automated tests and executable examples to support reliability across operator-learning and hybrid physics workflows. The test suite includes unit tests for the tracing DSL, trace evaluation, domain and geometry operations, derivative operators, model architectures, adaptive resampling, configuration management, signed save/load workflows, multi-device execution, and IREE export paths [@openxla2019iree]. Integration tests cover end-to-end training workflows with traced constraints, multi-model objectives, and solver-state serialization.

The repository uses `pytest` with custom markers to separate slow, integration, GPU, and serial test cases [@pytestdev2009pytest]. Fixtures in `conftest.py` provide deterministic RNG keys and mock domain objects for consistent test behavior. In addition to the automated test suite, the project includes tutorial and example scripts for validating representative training runs and expected output artifacts.

# Availability and reuse

jNO is implemented in Python and currently targets Python versions `>=3.11,<3.14`. Core dependencies include JAX, Equinox, Optax, PyGmsh, cloudpickle, and einops [@bradbury2018jax; @kidger2021equinox; @deepmind2020optax; @schlomer2018pygmsh; @cloudpipe2024cloudpickle; @arogozhnikov2022einops]. Optional extras provide CUDA-enabled JAX, development and testing tools, IREE integration, and foundation-model adapters.

The source repository is available at <https://github.com/FhG-IISB/jNO> under the MIT License. Related translated foundation-model repositories include Poseidon, Walrus, PDEformer2, MPP, and Morph. The package is also published on PyPI at <https://pypi.org/project/jNO/>.

jNO is reusable wherever researchers need to combine operator-learning models and PDE structure without maintaining separate code stacks. The same traced programming model supports paired-data operator learning, physics-informed regularization, mesh-based residual evaluation, and foundation-model fine-tuning. The project is also structured for extension: contributors can add models, operators, domain constructors, and resampling or tuning strategies without changing the core user-facing language.

# AI usage disclosure

GitHub Copilot with GPT-5.4 was used to assist with drafting and revising parts of the manuscript text and repository automation related to the paper submission workflow. The author reviewed, edited, and validated all AI-assisted output and remains fully responsible for the technical claims, wording, citations, and final submitted materials.

# Acknowledgements

I would like to thank my working students Rathan Ramesh and Janahvi Halgarkar for contributing to this project. I would also like to thank my team, Georg Kruse, Dr. Christopher Straub, Vlad Medvedev, Philipp Brendel, and Rodrigo Coehlo, for their input and guidance.

# Competing interests

The author declares that there are no competing interests.
