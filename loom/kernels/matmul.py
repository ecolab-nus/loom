"""Matmul kernel for the Loom pipeline.

Defines a helion matmul kernel and exposes generate_mlir() to produce
the stage-00 MLIR consumed by the loom-dataflow backend.

Kernel dimensions are a property of this kernel, not runtime arguments:
    M=4096, K=512, N=4096
"""

import torch
import helion
import helion.language as hl
from helion_mlir import generate_mlir as _helion_generate_mlir
from helion_mlir import print_debug_info


@helion.kernel(
    static_shapes=False,
    autotune_config_overrides={
        "range_unroll_factors": [0, 0],
        "range_num_stages": [0, 0],
    },
)
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


def generate_mlir() -> str:
    """Generate stage-00 MLIR for the matmul kernel.

    Side effect: prints debug info to stdout via print_debug_info.

    Returns:
        Complete MLIR text string (module { module { func.func @matmul ... } }).
    """
    x = torch.randn([4096, 512], device="cpu", dtype=torch.float32)
    y = torch.randn([512, 4096], device="cpu", dtype=torch.float32)
    bound_kernel = matmul.bind((x, y))
    print_debug_info(bound_kernel)
    return _helion_generate_mlir(bound_kernel)
