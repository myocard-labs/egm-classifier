"""PyTorch ``Dataset`` wrappers and the ``build_dataloaders`` convenience.

Torch is an optional dependency for ``myocard-egm-data``. Importing this
package directly triggers a torch import; if torch is not installed, the
ImportError is raised at import time with a hint to install
``myocard-egm-data[torch]``. Other subpackages (``banks``, ``records``,
``splits``, ``augmentation``) are torch-free.
"""

from __future__ import annotations

try:
    from .loaders import LoaderBundle, build_dataloaders
    from .trace_dataset import EGMTraceDataset
except ImportError as exc:  # pragma: no cover — torch absent
    raise ImportError(
        "myocard_egm_classifier.data.datasets requires PyTorch. Install with "
        "`pip install 'myocard-egm-data[torch]'`."
    ) from exc

__all__ = [
    "EGMTraceDataset",
    "LoaderBundle",
    "build_dataloaders",
]
