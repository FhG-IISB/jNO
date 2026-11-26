"""`input_dependencies`: which arguments a function's outputs can depend on for a derivative.

The FEM assemblers take each element block's tangent only against the fields its term reads, from this. A
position it leaves out gets NO column in the tangent, so it must never miss a real dependency: these tests
pin the cases that hide one (a jitted call, a loop, a custom derivative rule, a closure) and the one it may
cut (``stop_gradient``)."""

import jax
import jax.numpy as jnp

from jno.utils.solver.fem_utils import input_dependencies

a, b, c = jnp.ones(3), jnp.ones(2), jnp.ones(4)


def test_an_unused_argument_is_left_out():
    assert input_dependencies(lambda x, y, z: jnp.sin(x) * z[:3], a, b, c) == [0, 2]


def test_a_dependency_inside_a_jitted_call_is_found_and_an_unused_jit_input_is_not():
    inner = jax.jit(lambda x, y: jnp.sum(x) + 0.0 * jnp.sum(y[:1]))
    assert input_dependencies(lambda x, y, z: inner(x, y) * z, a, b, c) == [0, 1, 2]
    inner2 = jax.jit(lambda x, y: jnp.sum(x))
    assert input_dependencies(lambda x, y, z: inner2(x, y) * z, a, b, c) == [0, 2]


def test_a_loop_and_a_custom_rule_count_all_their_inputs():
    def loop(x, y, z):
        return jax.lax.fori_loop(0, 3, lambda i, s: s + jnp.sum(y), jnp.sum(x))

    assert input_dependencies(loop, a, b, c) == [0, 1]

    @jax.custom_jvp
    def f(x, y):
        return jnp.sum(x)

    f.defjvp(lambda p, t: (f(*p), jnp.sum(t[1])))
    assert input_dependencies(lambda x, y, z: f(x, y), a, b, c) == [0, 1]


def test_stop_gradient_cuts_the_dependency():
    assert input_dependencies(lambda x, y, z: jnp.sum(x) * jax.lax.stop_gradient(jnp.sum(y)), a, b, c) == [0]


def test_a_left_out_argument_has_a_zero_jacobian():
    """The contract the assemblers rely on, checked against jacfwd itself."""
    fn = lambda x, y, z: jnp.concatenate([jnp.sin(x) * jnp.sum(z), jnp.cos(z)])  # noqa: E731
    deps = input_dependencies(fn, a, b, c)
    J = jax.jacfwd(fn, argnums=(0, 1, 2))(a, b, c)
    for i, Ji in enumerate(J):
        assert (i in deps) or not jnp.any(Ji != 0)
