"""
Coupled PDE system with a manufactured solution
===============================================

We consider the following coupled elliptic PDE system on Ω ⊂ ℝ²:

    -Δu + v = f(x, y)
    -Δv + u = g(x, y)

Manufactured solution
---------------------
To enable quantitative verification, we prescribe an exact (manufactured)
solution:

    u(x, y) = sin(πx) sin(πy)
    v(x, y) = sin(2πx) sin(πy)

Source terms
------------
The corresponding source terms are obtained analytically by substituting
(u, v) into the PDE system:

    f(x, y) = 2π² sin(πx) sin(πy) + sin(2πx) sin(πy)
    g(x, y) = 5π² sin(2πx) sin(πy) + sin(πx) sin(πy)

The problem is solved using Physics-Informed Neural Networks (PINNs) with
a combination of hard constraints, soft constraints, and optional sensor data.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import jax

# Enable 64-bit floating point precision (float64) in JAX.
# This can improve numerical stability and, on some CPU workloads,
# may even be faster due to better vectorization and reduced
# accumulation error. On GPUs, however, float64 is usually slower
# and should be enabled only if higher precision is required.
jax.config.update("jax_enable_x64", False)  # or even better set $env:JAX_ENABLE_X64="True"
# note: this does not work with the soap optimzier at the moment (optax is fine)


import jno
import jno.numpy as jnn
from jno import LearningRateSchedule as lrs
from jno import WeightSchedule as ws
from jno import sampler

import jax.numpy as jnp
import optax
import equinox as eqx
from jno.architectures.linear import Linear
from soap_jax import soap

π = jnn.pi
sin = jnn.sin


dire = "./runs/coupled_system"


# Initialize the global logging instance at the top so that all the classes log to this file
log = jno.logger(dire)

# Custom Logs
log("Starting")

# -----------------------------------------------------------------------------
# Geometry and mesh construction
# -----------------------------------------------------------------------------


# To generate a mesh, we use pygmsh-style syntax.
# The function returns a constructor compatible with jno.domain.
def rect(x_range=(0, 1), y_range=(0, 1), mesh_size=0.1):
    def construct(geo):
        x0, x1 = x_range
        y0, y1 = y_range

        # Define the corner points of the rectangle
        points = [
            geo.add_point([x0, y0], mesh_size=mesh_size),
            geo.add_point([x1, y0], mesh_size=mesh_size),
            geo.add_point([x1, y1], mesh_size=mesh_size),
            geo.add_point([x0, y1], mesh_size=mesh_size),
        ]

        # Connect points with straight boundary segments
        lines = [
            geo.add_line(points[0], points[1]),
            geo.add_line(points[1], points[2]),
            geo.add_line(points[2], points[3]),
            geo.add_line(points[3], points[0]),
        ]

        # Create a surface enclosed by the boundary
        curve_loop = geo.add_curve_loop(lines)
        surface = geo.add_plane_surface(curve_loop)

        # Physical groups used for sampling and constraint definition
        geo.add_physical(surface, "interior")
        geo.add_physical(lines, "boundary")
        geo.add_physical([lines[0]], "bottom")
        geo.add_physical([lines[1]], "right")
        geo.add_physical([lines[2]], "top")
        geo.add_physical([lines[3]], "left")

        # Return geometry, spatial dimension, and characteristic mesh size
        return geo, 2, mesh_size

    return construct


# -----------------------------------------------------------------------------
# domain definition and sampling
# -----------------------------------------------------------------------------

# Construct the domain object and compute mesh-related quantities.
# These quantities are required for certain boundary conditions
# (e.g. view matrices, view factors).
domain = 2 * jno.domain(constructor=rect(mesh_size=0.1))
# domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.05))

# domains can be multiplied to 'tile' them -> making it easier to do operator learning on the same mesh

# Alternatively, an externally generated mesh can be loaded:
# domain = jno.domain('./runs/mesh.msh')

# Sample all points in the mesh tagged as "interior".
x, y, t = domain.variable("interior", (None, None))

# Additional geometric quantities can be extracted from the mesh.
# Here we request boundary points, outward normals, and view-related operators.
xb, yb, tb, nx, ny, VF = domain.variable("boundary", (None, None), normals=True, view_factor=True)

# Point-like quantities (e.g. sensor locations) can be added as arrays.
# The required shape is (B, N, dim).
xs, ys = domain.variable("sensor", 0.5 * jnp.ones((2, 1, 2)), point_data=True, split=True)

# Any other tensor-valued data can be attached to the domain.
# The minimal required dimension is (B, ...).
# Dimensions of shape 1 will be sqeezed out

k = domain.variable("k", jnp.array([[1.0], [1.0]]))  # shape = (2, 1)

# To plot the domain with its normals
domain.plot(f"{dire}/train_domain.png")


# -----------------------------------------------------------------------------
# Neural network models
# -----------------------------------------------------------------------------
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)


# Models are defined using equinox modules or jno-provided templates.
class MLP(eqx.Module):
    """Simple fully-connected neural network with scalar output."""

    dense1: Linear
    dense2: Linear
    dense3: Linear

    def __init__(self, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.dense1 = Linear(2, 64, key=k1)
        self.dense2 = Linear(64, 64, key=k2)
        self.dense3 = Linear(64, 1, key=k3)

    def __call__(self, x, y, k):

        # jax.debug.print("k has shape: {arr}", arr=k.shape)

        h = jnp.concat([x, y], axis=-1)
        h = jnp.tanh(self.dense1(h))
        h = jnp.tanh(self.dense2(h))
        u = self.dense3(h)
        return u


# Alternatively, wrap a custom equinox Module for v.
v_net = jnn.nn.wrap(MLP(key=k2))
v = v_net(x, y, k)


# Use a predefined MLP model for u.
# Multiplication by x(1-x)y(1-y) hard-enforces homogeneous Dirichlet BCs.
u_net = jnn.nn.mlp(2, hidden_dims=64, num_layers=2, key=k1)

u_net.dont_show()  # Do not print out the model

u = u_net(x, y) * x * (1 - x) * y * (1 - y)

# Wrapper for flax.debug.print -> use _shape, _min, _val, _max, _mean
# u.debug._shape = True


# The resulting neural networks can be combined freely
# with jno.numpy differential operators.

# -----------------------------------------------------------------------------
# Analytical (manufactured) solution
# -----------------------------------------------------------------------------

_u = sin(π * x) * sin(π * y)
_v = sin(2 * π * x) * sin(π * y)


# -----------------------------------------------------------------------------
# PDE residuals and constraints
# -----------------------------------------------------------------------------

# Define Laplacian operators.
# Although the names differ, both currently use finite differences.
Δfd = lambda inp: jnn.laplacian(inp, [x, y], scheme="finite_difference")
Δad = lambda inp: jnn.laplacian(inp, [x, y], scheme="automatic_differentiation")

# Interior PDE residuals
pde1 = -Δfd(u(x, y)) + v(x, y, k) - (2 * π**2 * _u(x, y) + _v(x, y))
pde2 = -Δad(v(x, y, k)) + u(x, y) - (5 * π**2 * _v(x, y) + _u(x, y))

# Softly enforced Dirichlet boundary condition for v.
# (u is enforced hard via the network architecture.)
boc2 = v(xb, yb, k) - 0.0

# Sensor constraint: enforce u = 1 at a specific interior location.
sens = u(xs, ys) - 1.0

# -----------------------------------------------------------------------------
# Debugging and monitoring utilities
# -----------------------------------------------------------------------------

# Using jax.debug.print inside BinaryOps allows inspection of intermediate values.
# This is extremely expensive and should only be enabled for debugging.
# Available options: (_val, _shape, _min, _max, _mean)
# Example:
# pde1.debug._shape = True

# Validation quantities can be tracked during training.
# They do not contribute to the loss.
# This also allows switching to purely data-driven training by
# disabling the physics constraints.
val1 = jnn.tracker(jnn.mean(u(x, y) - _u(x, y)), 100)
val2 = jnn.tracker(jnn.mean(v(x, y, k) - _v(x, y)), 100)


# -----------------------------------------------------------------------------
# Problem instantiation
# -----------------------------------------------------------------------------

# To use the mean square error one has to add it manually
# A mesh can be created for data or model parallelism
crux = jno.core(constraints=[pde1.mse, pde2.mse, boc2.mse, sens.mse, val1, val2], domain=domain, rng_seed=42, mesh=(len(jax.devices()), 1))


# -----------------------------------------------------------------------------
# Training configuration
# -----------------------------------------------------------------------------

# Learning-rate and weight schedules are callable:
#   learning_rate(epoch, individual_losses) -> scalar
#   weights(epoch, individual_losses) -> list of constraint weights
#
# Training is performed in optimized chunks.
# Frequent calls to .solve() should be avoided, as recompilation is expensive.
# After each solve call, a statistics object is returned and can be plotted.
#
# Checkpoints (params, optimizer state, RNG) are saved automatically after every .solve call

# Phase 1: Adam optimizer with warmup + cosine decay
u_net.optimizer(optax.adam, lr=lrs.warmup_cosine(4000, 500, 1e-3, 1e-4))
v_net.optimizer(optax.adam, lr=lrs.warmup_cosine(4000, 500, 1e-3, 1e-4))
crux.solve(4000, constraint_weights=ws([1.0, 1.0, 10.0, 1.0])).plot(f"{dire}/training_history_adam.png")


# Phase 1: Adam optimizer with gradient clipping
u_net.optimizer(optax.chain(optax.adam(1), optax.clip_by_global_norm(1e-3)), lr=lrs.warmup_cosine(4000, 500, 1e-3, 1e-4))
v_net.optimizer(optax.chain(optax.adam(1), optax.clip_by_global_norm(1e-3)), lr=lrs.warmup_cosine(4000, 500, 1e-3, 1e-4))
crux.solve(4000, constraint_weights=ws([1.0, 1.0, 10.0, 1.0])).plot(f"{dire}/training_history_adam_with_clip.png")


# Phase 2: SOAP optimizer with exponential decay and all weights are 1.0
u_net.optimizer(soap(1), lr=lrs(lambda e, _: 1e-4 * (5e-5 / 1e-4) ** (e / 1000)))
v_net.optimizer(soap(1), lr=lrs(lambda e, _: 1e-4 * (5e-5 / 1e-4) ** (e / 1000)))
crux.solve(1000).plot(f"{dire}/training_history_soap.png")


# Phase 3: L-BFGS refinement with adaptive boundary weighting
u_net.optimizer(optax.lbfgs, lr=lrs(5e-5))
v_net.optimizer(optax.lbfgs, lr=lrs(5e-5))
crux.solve(1000, constraint_weights=ws(lambda e, L: [1.0, 1.0, 10.0, 1.0 * L[3]])).plot(f"{dire}/training_history_lbfgs.png")

# -----------------------------------------------------------------------------
# Checkpointing and restart
# -----------------------------------------------------------------------------

# Save the entire training instance using cloudpickle.
crux.save(f"{dire}/trial.pkl")
del crux

# Reload the saved instance and continue training.
# By default, the last stored parameters are used.
crux = jno.core.load(f"{dire}/trial.pkl")
crux.set_optimizer(soap(1), lr=lrs(1e-5))
crux.solve(500, constraint_weights=ws([1.0, 1.0, 10.0, 10.0])).plot(f"{dire}/training_history_full_full.png")
