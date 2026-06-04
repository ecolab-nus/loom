"""Custom kernel for the Loom pipeline.

Standalone CLI script. Run from the repo root:

    python kernels/custom_op.py --config kernels/config_files/ --njobs 16 --debug --topk-candidates 1 --topk-block-size 3

This script inherits the full Loom CLI and pipeline from LoomKernel.
To write your own kernel, copy this file, replace the kernel body
and bind_args tensors, and keep the __main__ block unchanged.
"""


import torch
import helion
import helion.language as hl

from loom import LoomKernel


def _custom_op(
    input: torch.Tensor,
) -> torch.Tensor:
    return 


class CustomOp(LoomKernel):

    kernel_name = "Flash Attention"

    Size: int = 0

    # Assign the helion-decorated function as a class attribute.
    # We cannot stack @staticmethod with @helion.kernel because the helion
    # decorator returns a custom object, not a plain callable.
    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_custom_op)

    @classmethod
    def bind_args(cls) -> tuple:
        """Return concrete input tensors that define input at MLIR-gen time."""
        input = torch.empty([cls.Size], dtype=torch.float16)
        return (input)


if __name__ == "__main__":
    CustomOp.run()
