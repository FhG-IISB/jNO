from __future__ import annotations

import io
import pathlib
import struct
import sys
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Type, TypeVar, Union, overload

import cloudpickle

from ..core import core
from ..domain import domain
from .iree import IREEModel

TLoaded = TypeVar("TLoaded", core, domain, IREEModel)


def _rebuild_local_path(path_text: str):
    return Path(path_text)


def _rebuild_pure_path(path_text: str):
    return PurePath(path_text)


def _rebuild_pure_posix_path(path_text: str):
    return PurePosixPath(path_text)


def _rebuild_pure_windows_path(path_text: str):
    return PureWindowsPath(path_text)


def _reduce_path_for_pickle(path_obj: PurePath):
    class_name = type(path_obj).__name__
    if class_name == "PureWindowsPath":
        return _rebuild_pure_windows_path, (str(path_obj),)
    if class_name == "PurePosixPath":
        return _rebuild_pure_posix_path, (str(path_obj),)
    if class_name == "PurePath":
        return _rebuild_pure_path, (str(path_obj),)
    return _rebuild_local_path, (str(path_obj),)


class _CompatCloudPickler(cloudpickle.CloudPickler):
    dispatch_table = dict(cloudpickle.CloudPickler.dispatch_table)


for _path_type in {
    Path(".").__class__,
    PurePath(".").__class__,
    PurePosixPath,
    PureWindowsPath,
}:
    _CompatCloudPickler.dispatch_table[_path_type] = _reduce_path_for_pickle


def _compat_cloudpickle_dump(obj: Any, file_obj) -> None:
    _CompatCloudPickler(file_obj).dump(obj)


def _compat_cloudpickle_dumps(obj: Any) -> bytes:
    buffer = io.BytesIO()
    _compat_cloudpickle_dump(obj, buffer)
    return buffer.getvalue()


def _compat_cloudpickle_load_from_bytes(payload: bytes) -> Any:
    try:
        return cloudpickle.load(io.BytesIO(payload))
    except ModuleNotFoundError as exc:
        if exc.name != "pathlib._local":
            raise
        sys.modules.setdefault("pathlib._local", pathlib)
        return cloudpickle.load(io.BytesIO(payload))


class _CompatCloudpickleSerializer:
    CloudPickler = _CompatCloudPickler

    @staticmethod
    def dump(obj: Any, file_obj) -> None:
        _compat_cloudpickle_dump(obj, file_obj)

    @staticmethod
    def dumps(obj: Any) -> bytes:
        return _compat_cloudpickle_dumps(obj)

    @staticmethod
    def load(file_obj) -> Any:
        return _compat_cloudpickle_load_from_bytes(file_obj.read())

    @staticmethod
    def loads(payload: bytes) -> Any:
        return _compat_cloudpickle_load_from_bytes(payload)


def save(
    instance,
    filepath: str,
    public_key_path: str | None = None,
    private_key_path: str | None = None,
):
    """Save an object to a pickle file.

    If *public_key_path* / *private_key_path* are not provided, jNO checks
    whether RSA keys are configured in ``.jno.toml`` (or ``~/.jno/config.toml``)
    and uses them automatically.
    """
    from .config import get_rsa_private_key, get_rsa_public_key

    if public_key_path is None:
        public_key_path = get_rsa_public_key()
    if private_key_path is None:
        private_key_path = get_rsa_private_key()

    if public_key_path is not None and private_key_path is not None:
        try:
            from pylotte.signed_pickle import SignedPickle
        except ImportError as e:
            raise ImportError(
                "pylotte is required for signed save/load functionality. "
                "Install with `pip install pylotte` or `pip install jax-numerical-operators[dev]`"
            ) from e
        signer = SignedPickle(
            public_key_path=public_key_path,
            private_key_path=private_key_path,
            serializer=_CompatCloudpickleSerializer,
        )
        sig_path = f"{filepath.rsplit('.', 1)[0]}.sig"
        signer.dump_and_sign(instance, filepath, sig_path)
        instance.log.info(f"Signature saved to: {sig_path}")
    else:
        with open(filepath, "wb") as f:
            _compat_cloudpickle_dump(instance, f)

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
            raise ImportError(
                "pylotte is required for signed save/load functionality. "
                "Install with `pip install pylotte` or `pip install jax-numerical-operators[dev]`"
            ) from e
        loader = SignedPickle(public_key_path=public_key_path, serializer=_CompatCloudpickleSerializer)
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
            instance = _compat_cloudpickle_load_from_bytes(f.read())

    if not isinstance(instance, (core, domain, IREEModel)):
        raise TypeError(f"Loaded object has unsupported type: {type(instance).__name__}")

    if expected_type is not None:
        if not isinstance(instance, expected_type):
            raise TypeError(f"Expected {expected_type.__name__} from load(), got {type(instance).__name__}")
        return instance

    return instance
