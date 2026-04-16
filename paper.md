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
  - name: Rathan Ramesh
    affiliation: 1
  - name: Georg Kruse
    affiliation: 1
  - name: Christopher Straub
    affiliation: 1
affiliations:
  - name: Fraunhofer Institute for Integrated Systems and Device Technology IISB, Germany
    index: 1
date: 26 March 2026
bibliography: references.bib
---

# Summary

jNO (jax Neural Operators) is a JAX-native library for neural operators and PDE foundation-model workflows with unified support for data-driven and physics-informed training. Its core design is a tracing system in which domains, model calls, residuals, supervised losses, and diagnostics are written in one symbolic language and compiled into a single optimization pipeline [@bradbury2018jax].

The same user-facing interface supports operator regression, mesh-aware residual evaluation, and PDE-constrained training. jNO integrates mesh generation/loading, model composition, fine-grained training controls, and hyperparameter tuning, while keeping workflows JAX-native for efficient JIT-compiled execution.

# Statement of need

Neural operators are increasingly used in surrogate modeling, inverse problems, and PDE-constrained learning [@li2021fno; @lu2021deeponet]. In practice, software workflows are often fragmented across separate abstractions for geometry, model definition, loss construction, and training loops. This fragmentation is especially limiting when users combine supervised operator learning with physics-informed residuals or adapt pretrained backbones to new PDE settings.

jNO addresses this by treating the full workflow as one traced program: users define domains and variables, compose model and differential expressions, and solve through a single interface. This reduces orchestration overhead and supports hybrid objectives in one consistent execution model.

# State of the field

DeepXDE is a widely used equation-centric PINN framework [@lu2021deepxde]. JAX-PI provides a JAX-native PINN-focused alternative [@predictive2024jaxpi]. PhysicsNeMo and NeuralPDE.jl provide complementary ecosystems for physics-based scientific machine learning [@hennigh2021nvidia; @zubov2021neuralpde].

jNO differs by centering on one traced symbolic layer for neural operators and hybrid physics objectives in JAX. It also targets consolidation of translated PDE foundation-model families (including Poseidon, Walrus, PDEformer2, MPP, and Morph) into a shared JAX-native workflow for more reproducible comparison and transfer-learning pipelines [@fhgiisb2026jaxposeidon; @fhgiisb2026jaxwalrus; @fhgiisb2026jaxpdeformer2; @armbrusl2026jaxmpp; @armbrusl2026jaxmorph].

# Software design

The tracing layer is implemented as a symbolic DSL centered on `Placeholder` nodes. Arithmetic, slicing, reduction, and differential operators construct deferred expression trees instead of executing eagerly. Internally, symbolic nodes use identity-based equality/hashing and repeated callable fragments are normalized through operation-definition wrappers, which helps stable graph reuse across complex hybrid objectives. Before execution, jNO applies common sub-expression elimination so structurally identical subtrees (including model calls and derivative nodes) are shared, reducing redundant compile/runtime work.

Runtime evaluation is handled by a typed trace evaluator with explicit variable bindings per batch context. Differential operators support both automatic differentiation and finite-difference pathways, with finite-difference behavior driven by mesh-aware domain metadata. The tracing stack also exposes shape diagnostics and tracked expressions, which improves debuggability for multi-model or mixed residual/supervised programs.

The domain layer connects geometry, mesh tags, and training contexts. It supports mesh generation via PyGmsh/Gmsh and external mesh loading via meshio [@schlomer2018pygmsh; @geuzaine2009gmsh; @nschloe2024meshio]. Built-in geometries and physical tags (for example interior, boundary, and side-specific subsets) are stored alongside runtime context arrays, including separate temporal context handling. The same interface supports batched operator-learning setups by repeating/merging domains and attaching tensor tags for per-sample parameters.

For derivative and residual workflows, domain preprocessing provides connectivity metadata (neighbor/topology information, boundary indexing, and geometry-derived normals). jNO also includes adaptive point-resampling hooks so training can update sampling sets without changing symbolic equations. These capabilities allow one DSL to span mesh-aware residual learning and operator-learning pipelines with shared abstractions.

jNO extends this interface to variational workflows: weak forms are written in the same symbolic style and lowered to FEM/VPINN-related assembly backends with JAX-FEM integration [@xue2023jax]. Volume and tagged-boundary quadrature regions are represented as tagged domain variables, so weak-form and residual formulations remain consistent at the user API level.

Model architectures are exposed through one API, covering spectral, convolutional, attention-based, branch-trunk, and point-based operator families. Custom Equinox/Flax modules can be wrapped into the same tracing and optimization stack [@kidger2021equinox; @heek2024flax]. Models are configured with per-model optimizer attachment and parameter-level controls including freeze/unfreeze, masking, and LoRA-style adaptation [@hu2022lora].

At runtime, the core solver compiles traced constraints into a single JIT-compiled step with support for multi-device execution, explicit sharding, gradient checkpointing, data offloading, fused inner steps, and bounded profiling [@bradbury2018jax]. Hyperparameter tuning is integrated via `ArchSpace` (categorical/continuous/discrete spaces) with grid and Nevergrad-based search [@facebook2018nevergrad].

jNO provides persistence for solver/domain states and exported inference wrappers, using cloudpickle-based serialization and optional signed artifact workflows [@cloudpipe2024cloudpickle; @alpamayo2026pylotte]. Export paths include IREE-oriented inference wrappers for ahead-of-time deployment [@openxla2019iree].

# Research impact statement

jNO provides a reusable JAX-native software base for neural-operator and PDE-constrained learning workflows that are otherwise split across multiple tools. The project lowers reuse barriers through packaging/distribution, validates workflows through tests and executable examples, and supports foundation-model adaptation workflows through a unified traced interface.

By consolidating operator learning and physics-informed training in one symbolic execution model, jNO enables more consistent experimentation, transfer learning, and model comparison in the JAX ecosystem.

# Quality control

Quality assurance combines automated tests and executable examples. The test suite covers tracing/evaluation logic, domain and geometry utilities, derivative operators, model integrations, adaptive resampling, persistence utilities, and multi-device execution paths, with integration tests for end-to-end training workflows.

Testing uses `pytest` with markers for slow, integration, GPU, and serial scenarios [@pytestdev2009pytest]. Deterministic fixtures are used where appropriate for reproducibility.

# Availability and reuse

jNO is implemented in Python and targets versions `>=3.11,<3.14`. Core dependencies include JAX, Equinox, Optax, PyGmsh, cloudpickle, and einops [@bradbury2018jax; @kidger2021equinox; @deepmind2020optax; @schlomer2018pygmsh; @cloudpipe2024cloudpickle; @arogozhnikov2022einops]. Optional extras include CUDA-enabled JAX, development/testing tooling, and IREE support [@openxla2019iree].

Code repository: <https://github.com/FhG-IISB/jNO> (EPL-2.0 License). Related translated model repositories include Poseidon, Walrus, PDEformer2, MPP, and Morph. jNO is reusable for paired-data operator learning, physics-informed regularization, mesh-based residual workflows, and foundation-model fine-tuning.

# AI usage disclosure

GitHub Copilot was used to assist with drafting and revising parts of the manuscript text and repository automation related to the paper submission workflow. The author reviewed, edited, and validated all AI-assisted output and remains fully responsible for the final technical claims, wording, and citations.

# Acknowledgements

We thank members of the AI-Augmented Research Group, including Vlad Medvedev, Philipp Brendel, and Rodrigo Coehlo, for their input and guidance.

# Competing interests

The author declares that there are no competing interests.