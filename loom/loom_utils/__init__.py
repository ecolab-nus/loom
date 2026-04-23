"""Loom utility package.

Contains the pinned version of loom-dataflow that this loom codebase was
written against.  When loom-dataflow bumps its version for an API change,
update LOOM_DATAFLOW_REQUIRED_VERSION here AND adapt the calling code before
committing.
"""
from importlib.metadata import version as _pkg_version

# Pin the exact loom-dataflow version this code was written against.
# Bump this manually when updating loom to a new loom-dataflow API.
LOOM_DATAFLOW_REQUIRED_VERSION = "1.0.3"

_installed = _pkg_version("loom-dataflow")
if _installed != LOOM_DATAFLOW_REQUIRED_VERSION:
    raise RuntimeError(
        f"loom-dataflow version mismatch: loom requires "
        f"{LOOM_DATAFLOW_REQUIRED_VERSION}, but {_installed} is installed. "
        f"Either update loom to support {_installed} and bump "
        f"LOOM_DATAFLOW_REQUIRED_VERSION, or reinstall the expected version:\n"
        f"  pip install loom-dataflow=={LOOM_DATAFLOW_REQUIRED_VERSION}"
    )
