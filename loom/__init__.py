"""Loom — end-to-end ML kernel compilation pipeline.

Public API
----------
``LoomKernel``
    Base class for kernel scripts.  Inherit from this to get a full CLI
    + Loom compilation pipeline automatically.

``run_pipeline``
    Low-level entry point.  Pass a callable ``generate_mlir_fn() -> str``
    plus hardware paths to run the pipeline programmatically.

Example
-------
::

    from loom import LoomKernel

    class Matmul(LoomKernel):
        kernel = helion.kernel(...)(matmul_fn)

        @classmethod
        def bind_args(cls):
            return (torch.randn([4096, 512], dtype=torch.float16),
                    torch.randn([512, 4096], dtype=torch.float16))

    if __name__ == "__main__":
        Matmul.run()
"""

from .kernel_base import LoomKernel
from .pipeline import run_pipeline, setup_logging

__all__ = ["LoomKernel", "run_pipeline", "setup_logging"]
