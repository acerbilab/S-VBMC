# AI Summary: Public API exposing core functionality and metadata; wraps submodules for external use.
"""
Public package interface for **svbmc**.

Exports:
- `SVBMC` class
- `targets` and `utils` sub‑modules
- `__version__` string
"""
from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PkgNotFound

from .svbmc import SVBMC  # noqa: F401

from . import targets as targets  # re‑export
from . import utils as utils      # re‑export

try:
    __version__ = _pkg_version(__name__)
except _PkgNotFound:
    # Package is being used from source (not installed); fall back to internal constant.
    from .svbmc import __version__ as _src_version

    __version__ = _src_version  # type: ignore

__all__ = ["SVBMC", "targets", "utils", "__version__"]
