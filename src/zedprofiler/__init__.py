"""Top-level package for ZedProfiler.

Selected featurizers are promoted for convenient imports, while the full
module tree remains available under ``zedprofiler.featurization``.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from zedprofiler._version import __version__

if TYPE_CHECKING:
    from zedprofiler.featurization import colocalization, intensity

_PROMOTED_MODULES = {
    "colocalization": "zedprofiler.featurization.colocalization",
    "intensity": "zedprofiler.featurization.intensity",
}

__all__ = ["__version__", "colocalization", "intensity"]


def __getattr__(name: str) -> ModuleType:
    """Lazily load promoted modules on first access."""
    if name in _PROMOTED_MODULES:
        module = import_module(_PROMOTED_MODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module 'zedprofiler' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Include lazy exports in interactive completion."""
    return sorted(set(globals()) | set(__all__))
