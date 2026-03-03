"""Tests for encrypted (RSA-signed) save/load in core_utilities."""

import os

import jax
import pytest

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_rsa_keypair(directory: str):
    """Generate a 2048-bit RSA key pair and write PEM files to *directory*."""
    os.makedirs(directory, exist_ok=True)
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )
    public_key = private_key.public_key()

    private_path = os.path.join(directory, "private.pem")
    public_path = os.path.join(directory, "public.pem")

    with open(private_path, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    with open(public_path, "wb") as f:
        f.write(
            public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )

    return public_path, private_path


def _sig_path(filepath: str) -> str:
    """Mirror the sig_path derivation from core_utilities.save."""
    return f"{filepath.rsplit('.', 1)[0]}.sig"


def _make_solver():
    """Build and briefly train a minimal 1-D Laplace solver."""
    import optax
    import jno
    import jno.numpy as jnn
    from jno import LearningRateSchedule as lrs

    domain = jno.domain(constructor=jno.domain.line(mesh_size=0.05))
    x, _ = domain.variable("interior")

    key = jax.random.PRNGKey(42)
    u_net = jnn.nn.mlp(1, hidden_dims=16, num_layers=2, key=key)
    u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 100, 1e-5))
    u = u_net(x) * x * (1 - x)
    pde = jnn.laplacian(u, [x]) - jnn.sin(jnn.pi * x)

    solver = jno.core([pde.mse], domain)
    solver.solve(10)
    return solver


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEncryptedSaveLoad:

    def test_plain_save_and_load(self, tmp_path):
        """Save without keys and reload — plain cloudpickle round-trip."""
        import jno

        solver = _make_solver()
        pkl_path = str(tmp_path / "crux.pkl")

        jno.save(solver, pkl_path)

        assert os.path.exists(pkl_path)

        loaded = jno.load(pkl_path)
        assert loaded is not None
        assert hasattr(loaded, "models")

    def test_encrypted_save_produces_sig_file(self, tmp_path):
        """Saving with keys must write both the pickle and the .sig file."""
        import jno

        pub, priv = _generate_rsa_keypair(str(tmp_path / "keys"))
        solver = _make_solver()
        pkl_path = str(tmp_path / "crux_signed.pkl")
        sig = _sig_path(pkl_path)

        jno.save(solver, pkl_path, public_key_path=pub, private_key_path=priv)

        assert os.path.exists(pkl_path), "pickle file not written"
        assert os.path.exists(sig), f".sig file not written (expected {sig})"

    def test_load_with_correct_keys_succeeds(self, tmp_path):
        """Loading a signed file with the correct public key must succeed."""
        import jno

        pub, priv = _generate_rsa_keypair(str(tmp_path / "keys"))
        solver = _make_solver()
        pkl_path = str(tmp_path / "crux_signed.pkl")
        sig = _sig_path(pkl_path)

        jno.save(solver, pkl_path, public_key_path=pub, private_key_path=priv)
        loaded = jno.load(pkl_path, public_key_path=pub, signature_path=sig)

        assert loaded is not None
        assert hasattr(loaded, "models")

    def test_load_signed_file_without_keys_still_works(self, tmp_path):
        """A signed file can be loaded without key verification (plain fallback)."""
        import jno

        pub, priv = _generate_rsa_keypair(str(tmp_path / "keys"))
        solver = _make_solver()
        pkl_path = str(tmp_path / "crux_signed.pkl")

        jno.save(solver, pkl_path, public_key_path=pub, private_key_path=priv)

        # Load without providing keys — no signature check, just deserialise.
        loaded = jno.load(pkl_path)
        assert loaded is not None
        assert hasattr(loaded, "models")

    def test_load_with_wrong_public_key_raises(self, tmp_path):
        """Loading a signed file with a mismatched public key must raise."""
        import jno

        pub, priv = _generate_rsa_keypair(str(tmp_path / "keys"))
        wrong_pub, _ = _generate_rsa_keypair(str(tmp_path / "wrong_keys"))

        solver = _make_solver()
        pkl_path = str(tmp_path / "crux_signed.pkl")
        sig = _sig_path(pkl_path)

        jno.save(solver, pkl_path, public_key_path=pub, private_key_path=priv)

        with pytest.raises((ValueError, Exception)):
            jno.load(pkl_path, public_key_path=wrong_pub, signature_path=sig)
