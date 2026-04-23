"""Matmul kernel for the Loom pipeline.

Standalone CLI script. Run from the repo root:

    python kernels/matmul.py --config kernels/config_files/matmul.json --njobs 16 --debug --optimal-only

This script inherits the full Loom CLI and pipeline from LoomKernel.
To write your own kernel, copy this file, replace the kernel body
and bind_args tensors, and keep the __main__ block unchanged.
"""

import torch
import helion
import helion.language as hl

from loom import LoomKernel


def _matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Matrix-multiply x @ y using helion tiling.

    Kernel dimensions are determined at bind_args() time (M=4096, K=512, N=4096).
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float16)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


class Matmul(LoomKernel):
    """Matmul kernel: computes C = A @ B for fixed (M, K, N) shapes.

    Kernel dimensions (class-level constants, can be overridden in subclasses):
        M=4096, K=512, N=4096
    """

    kernel_name = "matmul"

    M: int = 4096
    K: int = 256
    N: int = 4096
    assume_divisible_tiles: bool = True

    # Assign the helion-decorated function as a class attribute.
    # We cannot stack @staticmethod with @helion.kernel because the helion
    # decorator returns a custom object, not a plain callable.
    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_matmul)

    @classmethod
    def bind_args(cls) -> tuple:
        """Return concrete input tensors that define M, K, N at MLIR-gen time."""
        x = torch.randn([cls.M, cls.K], device="cpu", dtype=torch.float16)
        y = torch.randn([cls.K, cls.N], device="cpu", dtype=torch.float16)
        return (x, y)


if __name__ == "__main__":
    Matmul.run()
