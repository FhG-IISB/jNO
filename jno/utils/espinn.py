# TODO Implement one


class espinn:

    def __init__(self, core_inst):
        self.core_inst = core_inst

    def alignment_matrix(self):

        compiled = self.core_inst.compiled
        layer_info = self.core_inst.layer_info
        domain_data = self.core_inst.domain_data
        config = self.core_inst.config

        def make_gradient_alignment_fn(compiled_constraints: List[Callable], layer_info: Dict, domain_data: DomainData, config: TrainingConfig):
            """Create function that computes gradient alignment matrix between constraints."""

            n_batches = domain_data.n_batches
            batch_size = min(config.batch_size, n_batches)
            ordered_tags = domain_data.ordered_tags
            points_arrays = domain_data.points_arrays
            tensor_tags = domain_data.tensor_tags
            n_constraints = len(compiled_constraints)

            def flatten_params(params: Dict) -> jax.Array:
                """Flatten parameter tree to vector."""
                leaves = jax.tree_util.tree_leaves(params)
                return jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])

            def gradient_alignment(params, rng):
                """Compute cosine similarity matrix between constraint gradients."""

                # Sample batch
                points_tuple = tuple(pts for pts in points_arrays)
                points_by_tag = dict(zip(ordered_tags, points_tuple))
                sliced_tensors = {k: v for k, v in tensor_tags.items()}

                # Compute gradient for each constraint
                def single_constraint_loss(params, idx):
                    fn = compiled_constraints[idx]
                    residual = fn(params, layer_info, points_by_tag, sliced_tensors)
                    return jnp.mean(jnp.square(residual))

                flat_grads = []
                for i in range(n_constraints):
                    grad = jax.grad(lambda p: single_constraint_loss(p, i))(params)
                    flat_grads.append(flatten_params(grad))

                # Stack and normalize
                grads_matrix = jnp.stack(flat_grads)  # (n_constraints, n_params)
                norms = jnp.linalg.norm(grads_matrix, axis=1, keepdims=True)
                normalized = grads_matrix / jnp.maximum(norms, 1e-8)

                # Cosine similarity matrix
                alignment_matrix = jnp.dot(normalized, normalized.T)

                lox.log({"alignment_matrix": alignment_matrix})

                return alignment_matrix

            return gradient_alignment

        alignment_fn = make_gradient_alignment_fn(compiled, layer_info, domain_data, config)
        alignment_matrix = jax.jit(lox.spool(alignment_fn))

        return alignment_matrix
