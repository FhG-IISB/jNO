from __future__ import annotations
import struct
import cloudpickle
from ..core import core
from ..domain import domain
from .iree import IREEModel
from typing import Union, TypeVar, Type, overload, Any


TLoaded = TypeVar("TLoaded", core, domain, IREEModel)


def save(instance, filepath: str, public_key_path: str | None = None, private_key_path: str | None = None):
    """Save an object to a pickle file.

    If *public_key_path* / *private_key_path* are not provided, jNO checks
    whether RSA keys are configured in ``.jno.toml`` (or ``~/.jno/config.toml``)
    and uses them automatically.
    """
    from .config import get_rsa_public_key, get_rsa_private_key

    if public_key_path is None:
        public_key_path = get_rsa_public_key()
    if private_key_path is None:
        private_key_path = get_rsa_private_key()

    if public_key_path is not None and private_key_path is not None:
        try:
            from pylotte.signed_pickle import SignedPickle
        except ImportError as e:
            raise ImportError("pylotte is required for signed save/load functionality. " "Install with `pip install pylotte` or `pip install jax-neural-operators[dev]`") from e
        signer = SignedPickle(
            public_key_path=public_key_path,
            private_key_path=private_key_path,
            serializer=cloudpickle,
        )
        sig_path = f"{filepath.rsplit('.', 1)[0]}.sig"
        signer.dump_and_sign(instance, filepath, sig_path)
        instance.log.info(f"Signature saved to: {sig_path}")
    else:

        with open(filepath, "wb") as f:
            cloudpickle.dump(instance, f)

    instance.log.info(f"Model/Domain saved to: {filepath}")
    return None


@overload
def load(
    filepath: str,
    public_key_path: str | None = None,
    signature_path: str | None = None,
    *,
    expected_type: Type[TLoaded],
) -> TLoaded: ...


@overload
def load(
    filepath: str,
    public_key_path: str | None = None,
    signature_path: str | None = None,
    *,
    expected_type: None = None,
) -> Union[core, domain, IREEModel]: ...


def load(
    filepath: str,
    public_key_path: str | None = None,
    signature_path: str | None = None,
    *,
    expected_type: Type[TLoaded] | None = None,
) -> Union[core, domain, IREEModel, TLoaded]:
    """Load a pickle object.

    If *public_key_path* is not provided, jNO checks whether an RSA public
    key is configured in ``.jno.toml`` (or ``~/.jno/config.toml``) and uses
    it automatically when a *signature_path* is supplied.
    """
    from .config import get_rsa_public_key

    if public_key_path is None and signature_path is not None:
        public_key_path = get_rsa_public_key()
    if public_key_path is not None and signature_path is not None:
        try:
            from pylotte.signed_pickle import SignedPickle
        except ImportError as e:
            raise ImportError("pylotte is required for signed save/load functionality. " "Install with `pip install pylotte` or `pip install jax-neural-operators[dev]`") from e
        loader = SignedPickle(public_key_path=public_key_path, serializer=cloudpickle)
        instance = loader.safe_load(filepath, signature_path)
    else:
        _MAGIC = b"PYLOTTE-SP\x01"
        with open(filepath, "rb") as f:
            prefix = f.read(len(_MAGIC))
            if prefix == _MAGIC:
                # Skip the pylotte header (4-byte length + JSON) so that
                # cloudpickle reads only the serialised payload.
                (length,) = struct.unpack(">I", f.read(4))
                f.read(length)
            else:
                f.seek(0)
            instance = cloudpickle.load(f)

    if not isinstance(instance, (core, domain, IREEModel)):
        raise TypeError(f"Loaded object has unsupported type: {type(instance).__name__}")

    if expected_type is not None:
        if not isinstance(instance, expected_type):
            raise TypeError(f"Expected {expected_type.__name__} from load(), got {type(instance).__name__}")
        return instance

    return instance
