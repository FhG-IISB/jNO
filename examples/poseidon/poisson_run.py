import jno
import optax
import argparse
import jno.numpy as jnn
from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
from jax import random
from jno import LearningRateSchedule as lrs
import os


class SourceTermGeneratorPoisson:
    """
    JAX-compatible generator for various source terms with distinct shapes.
    All source terms are scaled to amplitude range [0, max_amplitude].
    """

    def __init__(self, shape: Tuple[int, int] = (128, 128), max_amplitude: float = 1.5):
        self.shape = shape
        self.max_amplitude = max_amplitude

        # Precompute grids (static, so can be stored)
        x = jnp.linspace(0, 1, shape[0])
        y = jnp.linspace(0, 1, shape[1])
        self.X, self.Y = jnp.meshgrid(x, y)
        self.R = jnp.sqrt(self.X**2 + self.Y**2)
        self.Theta = jnp.arctan2(self.Y, self.X)

        # Precompute Gaussian kernels for smoothing (fixed size, stacked)
        self.n_kernels = 10
        self.kernel_size = 19  # Fixed size to accommodate largest sigma
        self._gaussian_kernels = self._precompute_gaussian_kernels()

    def _precompute_gaussian_kernels(self, max_sigma: float = 3.0):
        """Precompute Gaussian kernels for different sigma values with fixed size."""
        kernels = []
        for i in range(self.n_kernels):
            sigma = 0.5 + (max_sigma - 0.5) * i / (self.n_kernels - 1)
            ax = jnp.arange(self.kernel_size) - self.kernel_size // 2
            xx, yy = jnp.meshgrid(ax, ax)
            kernel = jnp.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            kernel = kernel / kernel.sum()
            kernels.append(kernel)
        return jnp.stack(kernels)  # Shape: (n_kernels, kernel_size, kernel_size)

    @partial(jax.jit, static_argnums=(0,))
    def _gaussian_filter(self, field: jnp.ndarray, sigma_idx: jnp.ndarray) -> jnp.ndarray:
        """Apply Gaussian filter using precomputed kernel."""
        # Ensure sigma_idx is within bounds
        sigma_idx = sigma_idx % self.n_kernels
        kernel = self._gaussian_kernels[sigma_idx]

        # Pad and convolve
        pad_size = self.kernel_size // 2
        padded = jnp.pad(field, pad_size, mode="edge")
        result = jax.scipy.signal.convolve2d(padded, kernel, mode="valid")
        return result

    @partial(jax.jit, static_argnums=(0,))
    def normalize(self, field: jnp.ndarray) -> jnp.ndarray:
        """Normalize field to [0, max_amplitude] range."""
        field_min = field.min()
        field_max = field.max()
        field = jnp.where(field_max - field_min > 1e-10, (field - field_min) / (field_max - field_min), field - field_min)
        return field * self.max_amplitude

    @partial(jax.jit, static_argnums=(0,))
    def gaussian_blobs(self, key: random.PRNGKey) -> jnp.ndarray:
        """Gaussian blob source term with random number of blobs (0-10)."""
        key_n, key_blobs = random.split(key)

        # Random number of blobs between 0 and 10
        n_blobs = random.randint(key_n, shape=(), minval=0, maxval=11)  # maxval is exclusive

        # Always generate keys for max blobs (10)
        max_blobs = 10
        keys = random.split(key_blobs, max_blobs * 5)
        blob_keys = keys.reshape(max_blobs, 5, 2)

        def add_blob(carry, inputs):
            field, idx = carry
            blob_key = inputs

            k_cx, k_cy, k_sx, k_sy, k_amp = blob_key

            cx = random.uniform(k_cx, minval=-0.8, maxval=0.8)
            cy = random.uniform(k_cy, minval=-0.8, maxval=0.8)
            sx = random.uniform(k_sx, minval=0.05, maxval=0.3)
            sy = random.uniform(k_sy, minval=0.05, maxval=0.3)
            amplitude = random.uniform(k_amp, minval=0.0, maxval=self.max_amplitude)

            gaussian = amplitude * jnp.exp(-((self.X - cx) ** 2 / (2 * sx**2) + (self.Y - cy) ** 2 / (2 * sy**2)))

            # Only add blob if idx < n_blobs
            field = jnp.where(idx < n_blobs, field + gaussian, field)

            return (field, idx + 1), None

        (field, _), _ = jax.lax.scan(add_blob, (jnp.zeros(self.shape), 0), blob_keys)

        return jnp.clip(field, 0, self.max_amplitude)

    @partial(jax.jit, static_argnums=(0,))
    def sinusoidal_waves(self, key: random.PRNGKey) -> jnp.ndarray:
        """Sinusoidal wave patterns with random orientations and frequencies."""
        n_waves = 4
        keys = random.split(key, n_waves * 3 + 1)

        def add_wave(carry, wave_keys):
            field = carry
            k_angle, k_freq, k_phase = wave_keys

            angle = random.uniform(k_angle, minval=0.0, maxval=2 * jnp.pi)
            freq = random.uniform(k_freq, minval=2.0, maxval=8.0)
            phase = random.uniform(k_phase, minval=0.0, maxval=2 * jnp.pi)

            wave = jnp.sin(freq * (self.X * jnp.cos(angle) + self.Y * jnp.sin(angle)) + phase)
            return field + wave, None

        wave_keys = keys[1:].reshape(n_waves, 3, 2)
        field, _ = jax.lax.scan(add_wave, jnp.zeros(self.shape), wave_keys)

        return self.normalize(jnp.abs(field))

    @partial(jax.jit, static_argnums=(0,))
    def radial_patterns(self, key: random.PRNGKey) -> jnp.ndarray:
        """Radial patterns (rings, gradient, bullseye)."""
        k1, k2, k3 = random.split(key, 3)

        pattern_idx = random.randint(k1, (), 0, 3)
        n_rings = random.randint(k2, (), 3, 10)
        power = random.uniform(k3, minval=0.5, maxval=3.0)

        # Compute all patterns and select
        rings = jnp.sin(self.R * n_rings * jnp.pi)
        gradient = 1 - self.R**power
        bullseye = jnp.cos(self.R * n_rings * jnp.pi)

        field = jnp.where(pattern_idx == 0, rings, jnp.where(pattern_idx == 1, gradient, bullseye))

        return self.normalize(jnp.maximum(field, 0))

    @partial(jax.jit, static_argnums=(0,))
    def angular_patterns(self, key: random.PRNGKey) -> jnp.ndarray:
        """Angular/spoke patterns."""
        k1, k2, k3 = random.split(key, 3)

        n_spokes = random.randint(k1, (), 3, 12)
        pattern_idx = random.randint(k2, (), 0, 2)
        decay_scale = random.uniform(k3, minval=0.5, maxval=2.0)

        sharp = jnp.abs(jnp.sin(n_spokes * self.Theta))
        smooth = (jnp.sin(n_spokes * self.Theta) + 1) / 2

        field = jnp.where(pattern_idx == 0, sharp, smooth)

        decay = jnp.exp(-self.R**2 / decay_scale)
        field = field * decay

        return self.normalize(field)

    @partial(jax.jit, static_argnums=(0,))
    def checkerboard_patterns(self, key: random.PRNGKey) -> jnp.ndarray:
        """Checkerboard or grid patterns with varying scales."""
        k1, k2 = random.split(key)

        n_squares = random.randint(k1, (), 4, 16)
        sigma_idx = random.randint(k2, (), 0, self.n_kernels)

        check_x = jnp.floor(self.X * n_squares / 2)
        check_y = jnp.floor(self.Y * n_squares / 2)
        field = ((check_x + check_y) % 2).astype(jnp.float32)

        field = self._gaussian_filter(field, sigma_idx)

        return self.normalize(field)

    @partial(jax.jit, static_argnums=(0,))
    def perlin_noise(self, key: random.PRNGKey) -> jnp.ndarray:
        """Perlin-like noise using octaves of sine waves."""
        n_octaves = 5
        keys = random.split(key, n_octaves * 2)

        def add_octave(carry, inputs):
            field, octave = carry
            k_x, k_y = inputs

            freq = 2.0**octave
            amplitude = 1.0 / (2.0**octave)

            phase_x = random.uniform(k_x, minval=0.0, maxval=100.0)
            phase_y = random.uniform(k_y, minval=0.0, maxval=100.0)

            noise = amplitude * (jnp.sin(freq * self.X + phase_x) * jnp.sin(freq * self.Y + phase_y))
            return (field + noise, octave + 1), None

        octave_keys = keys.reshape(n_octaves, 2, 2)
        (field, _), _ = jax.lax.scan(add_octave, (jnp.zeros(self.shape), 0), octave_keys)

        return self.normalize(jnp.abs(field))

    @partial(jax.jit, static_argnums=(0,))
    def stripe_patterns(self, key: random.PRNGKey) -> jnp.ndarray:
        """Parallel stripes with varying orientations and widths."""
        k1, k2, k3, k4, k5 = random.split(key, 5)

        angle = random.uniform(k1, minval=0.0, maxval=jnp.pi)
        n_stripes = random.randint(k2, (), 5, 15)
        pattern_idx = random.randint(k3, (), 0, 3)
        wave_freq = random.uniform(k4, minval=3.0, maxval=8.0)
        wave_amp = random.uniform(k5, minval=0.1, maxval=0.3)

        rotated = self.X * jnp.cos(angle) + self.Y * jnp.sin(angle)

        # Sharp stripes (smoothed) - use fixed sigma_idx for consistency
        sharp = (jnp.sin(rotated * n_stripes * jnp.pi) > 0).astype(jnp.float32)
        sharp = self._gaussian_filter(sharp, jnp.array(2))

        # Gradient stripes
        gradient = (jnp.sin(rotated * n_stripes * jnp.pi) + 1) / 2

        # Wavy stripes
        wavy = rotated + wave_amp * jnp.sin(wave_freq * (self.X * jnp.sin(angle) - self.Y * jnp.cos(angle)))
        wavy_field = (jnp.sin(wavy * n_stripes * jnp.pi) + 1) / 2

        field = jnp.where(pattern_idx == 0, sharp, jnp.where(pattern_idx == 1, gradient, wavy_field))

        return self.normalize(field)

    @partial(jax.jit, static_argnums=(0,))
    def spiral_patterns(self, key: random.PRNGKey) -> jnp.ndarray:
        """Spiral patterns (Archimedean or logarithmic)."""
        k1, k2, k3, k4 = random.split(key, 4)

        spiral_idx = random.randint(k1, (), 0, 2)
        n_arms = random.randint(k2, (), 2, 6)
        arch_scale = random.uniform(k3, minval=10.0, maxval=30.0)
        log_scale = random.uniform(k4, minval=2.0, maxval=5.0)

        archimedean = jnp.sin(n_arms * self.Theta - self.R * arch_scale)
        logarithmic = jnp.sin(n_arms * self.Theta - log_scale * jnp.log(self.R + 0.1))

        spiral = jnp.where(spiral_idx == 0, archimedean, logarithmic)

        envelope_scale = random.uniform(k4, minval=1.0, maxval=3.0)
        envelope = jnp.exp(-self.R**2 / envelope_scale)
        field = spiral * envelope

        return self.normalize(jnp.maximum(field, 0))

    @partial(jax.jit, static_argnums=(0, 2))
    def random_lines(self, key: random.PRNGKey, n_lines: int = 6) -> jnp.ndarray:
        """Multiple intersecting lines with Gaussian profiles."""
        keys = random.split(key, n_lines * 3)

        def add_line(field, line_keys):
            k_angle, k_offset, k_width = line_keys

            angle = random.uniform(k_angle, minval=0.0, maxval=jnp.pi)
            offset = random.uniform(k_offset, minval=-0.8, maxval=0.8)
            width = random.uniform(k_width, minval=0.02, maxval=0.1)

            dist = jnp.abs(self.X * jnp.sin(angle) - self.Y * jnp.cos(angle) - offset)
            line = jnp.exp(-(dist**2) / (2 * width**2))
            return field + line, None

        line_keys = keys.reshape(n_lines, 3, 2)
        field, _ = jax.lax.scan(add_line, jnp.zeros(self.shape), line_keys)

        return self.normalize(jnp.clip(field, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def random_polygons(self, key: random.PRNGKey, n_polygons: int = 5) -> jnp.ndarray:
        """Multiple polygons approximated with smooth shapes."""
        keys = random.split(key, n_polygons * 5 + 1)

        def add_polygon(field, poly_keys):
            k_sides, k_cx, k_cy, k_radius, k_rotation = poly_keys

            n_sides = random.randint(k_sides, (), 3, 8)
            cx = random.uniform(k_cx, minval=-0.7, maxval=0.7)
            cy = random.uniform(k_cy, minval=-0.7, maxval=0.7)
            radius = random.uniform(k_radius, minval=0.1, maxval=0.4)
            rotation = random.uniform(k_rotation, minval=0.0, maxval=2 * jnp.pi)

            # Approximate polygon with superellipse
            local_theta = jnp.arctan2(self.Y - cy, self.X - cx) - rotation
            local_r = jnp.sqrt((self.X - cx) ** 2 + (self.Y - cy) ** 2)

            # Polygon approximation using Fourier
            poly_r = radius * (1 + 0.1 * jnp.cos(n_sides * local_theta))
            mask = jnp.exp(-((local_r / poly_r) ** 4))

            return field + mask, None

        poly_keys = keys[1:].reshape(n_polygons, 5, 2)
        field, _ = jax.lax.scan(add_polygon, jnp.zeros(self.shape), poly_keys)

        sigma_idx = random.randint(keys[0], (), 0, self.n_kernels)
        field = self._gaussian_filter(field, sigma_idx)

        return self.normalize(field)

    @partial(jax.jit, static_argnums=(0,))
    def generate_random(self, key: random.PRNGKey) -> jnp.ndarray:
        """Generate a random source term from all available patterns."""
        k1, k2 = random.split(key)

        # Select pattern index
        pattern_idx = random.randint(k1, (), 0, 10)

        # Use lax.switch for efficient pattern selection
        def pattern_0(k):
            return self.sinusoidal_waves(k)

        def pattern_1(k):
            return self.radial_patterns(k)

        def pattern_2(k):
            return self.angular_patterns(k)

        def pattern_3(k):
            return self.checkerboard_patterns(k)

        # def pattern_4(k):
        #    return self.random_polygons(k)

        def pattern_5(k):
            return self.perlin_noise(k)

        def pattern_6(k):
            return self.stripe_patterns(k)

        def pattern_7(k):
            return self.spiral_patterns(k)

        def pattern_8(k):
            return self.random_lines(k)

        def pattern_9(k):
            return self.gaussian_blobs(k)

        # return jax.lax.switch(pattern_idx, [pattern_0, pattern_1, pattern_2, pattern_3, pattern_4, pattern_5, pattern_6, pattern_7, pattern_8, pattern_9], k2)
        return jax.lax.switch(pattern_idx, [pattern_0, pattern_1, pattern_2, pattern_3, pattern_5, pattern_6, pattern_7, pattern_8, pattern_9], k2)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune foundation models for physics-informed learning")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        choices=["poseidonT", "poseidonB", "poseidonL", "walrus", "cno", "pit", "fno", "unet", "pdeformer2_small", "pdeformer2_base", "pdeformer2_fast", "morphTi", "morphS", "morphM", "morphL", "mppTi", "mppS", "mppB", "mppL"],
    )
    parser.add_argument("--samples", "-s", type=int, help="Number of samples [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]")
    parser.add_argument("--training", "-t", type=str, choices=["data", "phys", "both"])
    parser.add_argument("--lora_rank", "-l", type=int, default=0, choices=[1, 2, 4, 8, 16, 32, 64, 128])
    args = parser.parse_args()

    EPOCHS = 800
    GENF = SourceTermGeneratorPoisson(max_amplitude=2.5)
    MODEL = args.model
    SAMPLES = args.samples
    TYPE = args.training
    LORA = True if args.lora_rank != 0 else False
    LORARANK = int(args.lora_rank)
    DATA_DRIVEN = True if TYPE in ["data", "both"] else False
    PHYS_INFORM = True if TYPE in ["phys", "both"] else False
    BATCHSIZE = 8  # SAMPLES if SAMPLES <= 32 else 40
    INFERENCE_PARAMS = SAMPLES if SAMPLES <= 32 else BATCHSIZE
    constraints = []

    PATH = os.path.dirname(os.path.abspath(__file__))
    l_path = f"{PATH}/runs/{MODEL}_{TYPE}_{SAMPLES}_lora{LORARANK}" if LORA else f"{PATH}/runs/{MODEL}_{TYPE}_{SAMPLES}"
    d_path = f"{PATH}/domain_{SAMPLES}_walrus_v2.pkl"
    if MODEL == "walrus":
        w_path = os.environ.get("WALRUS_WEIGHTS", "/home/b8cl/projects/DATA/walrus/walrus_base/walrus.msgpack")
    elif "morph" in MODEL:
        # morph variant name: "morphTi" → "Ti", "morphM" → "M", etc.
        morph_variant = MODEL[len("morph") :]
        w_path = os.environ.get("MORPH_WEIGHTS", f"/home/b8cl/projects/jax_morph/morph-{morph_variant}.msgpack")
    elif "mpp" in MODEL:
        mpp_variant = MODEL[len("mpp") :]  # "mppTi" → "Ti"
        w_path = os.environ.get("MPP_WEIGHTS", f"/home/b8cl/projects/DATA/mpp/mpp-{mpp_variant}.msgpack")
    else:
        w_path = os.environ.get("POSEIDON_WEIGHTS", f"/home/b8cl/projects/DATA/poseidon/{MODEL}.msgpack")
    jno.logger(l_path)

    domain = jno.domain.load(d_path)
    x, y, t = domain.variable("interior")
    _u = domain.variable("_u")  # (S, 1, 1, 128, 128, 1)
    _f = domain.variable("_f")  # (S, 1, 1, 128, 128, 1)
    # import h5py
    # import numpy as np

    # with h5py.File(f"/home/b8cl/projects/DATA/Poisson-Gauss.nc", "r") as file:
    #    _F = file["source"][...]  # (20000, 128, 128) or # (20000, 128, 128, 1)
    #    _U = file["solution"][...]  # (20000, 128, 128) or # (20000, 128, 128, 1)
    # time = np.ones((SAMPLES, 1))  # All 1's since it is a time-indepenent problem
    # _f = _F[240 : 240 + SAMPLES]
    # _u = _U[240 : 240 + SAMPLES]
    # domain = SAMPLES * jno.domain(jno.domain.poseidon(), compute_mesh_connectivity=True)
    ## x, y, t = domain.variable("interior")
    ## xb, yb  = domain.variable("boundary")
    # domain.variable("_f", _f[:, None, None, :, :, None])
    # domain.variable("_u", _u[:, None, None, :, :, None])
    # domain.save(d_path)

    # C = jnn.constant("C", {"mean_source": 0.014822142414492256, "std_source": 4.755138816607612, "mean_solution": 0.0005603458434937093, "std_solution": 0.02401226126952699})

    _u = (_u - jnn.mean(domain.context["_u"])) / jnn.std(domain.context["_u"])
    _f = (_f - jnn.mean(domain.context["_f"])) / jnn.std(domain.context["_f"])

    key = jax.random.PRNGKey(0)

    if "pdeformer2" in MODEL:
        import numpy as np
        import sys

        sys.path.insert(0, "/home/b8cl/projects/jax_pdeformer2/scripts")
        from compare import PDENodesCollector, x_fenc, y_fenc

        # Build static Poisson DAG:  u_xx + u_yy + f = 0,  u=0 on boundary
        pde = PDENodesCollector()
        u_var = pde.new_uf()
        # Source term f (coefficient function — placeholder zeros; structure is what matters)
        pde._add_func(np.zeros_like(x_fenc), x=x_fenc, y=y_fenc)
        f_node = pde._add_node("cf")
        # Zero Dirichlet BC
        pde._add_func(np.zeros_like(x_fenc), x=x_fenc, y=y_fenc)
        pde._add_node("bv", [u_var])
        # PDE equation:  u_xx + u_yy + f = 0
        u_xx = pde.dx(pde.dx(u_var))
        u_yy = pde.dy(pde.dy(u_var))
        pde.sum_eq0(u_xx, u_yy, f_node)
        dag = pde.gen_dag(uf_num_mod=11)

        pdeformer2_factory = {
            "pdeformer2_small": jnn.nn.pdeformer2_small,
            "pdeformer2_base": jnn.nn.pdeformer2_base,
            "pdeformer2_fast": jnn.nn.pdeformer2_fast,
        }
        variant = MODEL.split("_", 1)[1]  # "small" / "base" / "fast"
        w_path = os.environ.get("PDEFORMER2_WEIGHTS", f"/home/b8cl/projects/DATA/pdeformer/pdeformer2-{variant}.msgpack")
        u = pdeformer2_factory[MODEL](dag_inputs=dag).dont_show()
        u.initialize(w_path)
        # coordinate: [t=0, x, y, z=0] → (N, 4)
        coord = jnn.concat([x * 0.0, x, y, x * 0.0])
        input = [coord]

    elif MODEL == "fno":
        u = jnn.nn.fno2d(1, hidden_channels=48, n_modes=24, d_vars=1, n_layers=4, n_steps=1, d_model=(128, 128), key=key).dont_show()
        input = [_f]
    elif MODEL == "cno":
        u = jnn.nn.cno2d(size=128, key=key)
        input = [_f]
    elif MODEL == "pit":
        u = jnn.nn.pit(1, 1, n_head=2, input_res=(128, 128), output_res=(128, 128), key=key)
        input = [_f[0, ...].reshape((128 * 128, 1))]
    elif MODEL == "unet":
        u = jnn.nn.unet2d(in_channels=1, out_channels=1, depth=4, wf=6, key=key)
        input = [_f[0, ...]]
    elif MODEL == "walrus":
        u = jnn.nn.walrus((1, 1, 128, 128, 1), num_out_channels=1).dont_show()
        u.initialize(w_path)
        input = [_f]
    elif "morph" in MODEL:
        # MorphAdapter (built into the wrapper) reshapes jNO's per-sample
        # (1, H, W, 1) → MORPH's (B, t, F, C, D, H, W) = (1,1,1,1,1,128,128)
        # and maps output (1,F,C,D,H,W) back to (1, H, W, 1).
        morph_factory = {
            "morphTi": jnn.nn.morphTi,
            "morphS": jnn.nn.morphS,
            "morphM": jnn.nn.morphM,
            "morphL": jnn.nn.morphL,
        }
        u = morph_factory[MODEL](spatial_size=128).dont_show()
        u.initialize(w_path)
        input = [_f]
    elif "mpp" in MODEL:
        mpp_factory = {
            "mppTi": jnn.nn.mppTi,
            "mppS": jnn.nn.mppS,
            "mppB": jnn.nn.mppB,
            "mppL": jnn.nn.mppL,
        }
        u = mpp_factory[MODEL](spatial_size=128).dont_show()
        u.initialize(w_path)
        input = [_f]
    elif "poseidon" in MODEL:
        poseidon_factory = {"poseidonT": jnn.nn.poseidonT, "poseidonB": jnn.nn.poseidonB, "poseidonL": jnn.nn.poseidonL}
        u = poseidon_factory[MODEL](num_in_channels=1, num_out_channels=1).dont_show()
        u.initialize(w_path)
        input = [_f, t]

    # u.dtype(jnp.bfloat16)

    if DATA_DRIVEN:
        constraints.append(_u - u(*input))
    if PHYS_INFORM:
        constraints.append(jnn.laplacian(u(*input), [x, y], scheme="finite_difference") + _f)

    crux = jno.core([con.mse for con in constraints], domain, rng_seed=42, mesh=(len(jax.devices()), 1))
    crux.print_shapes()

    # Per-model optimizer + optional LoRA
    if LORA:
        u.lora(rank=LORARANK, alpha=LORARANK * 2.0)
    u.optimizer(
        optax.chain(optax.clip_by_global_norm(1e-3), optax.adamw(1.0, weight_decay=1e-6)),
        lr=lrs.cosine(EPOCHS, 5e-4, 1e-7),
    )

    crux.solve(epochs=EPOCHS, batchsize=BATCHSIZE, checkpoint_gradients=True, offload_data=True).plot(f"{l_path}/training_history.png")

    crux.domain = None
    crux.save(f"{l_path}/crux.pkl")

    return None


if __name__ == "__main__":
    main()
