"""Flash Attention kernel for the Loom pipeline.

Standalone CLI script. Run from the repo root:

    python kernels/flash_attention.py --config kernels/config_files/flash_attention.json --njobs 16 --debug

This script inherits the full Loom CLI and pipeline from LoomKernel.
To write your own kernel, copy this file, replace the kernel body
and bind_args tensors, and keep the __main__ block unchanged.
"""

import math
import torch
import helion
import helion.language as hl

from loom import LoomKernel
from helion_mlir.custom_op import broadcast


def _flash__attention(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out_ = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        qk_scale_dev = hl.full([], sm_scale, dtype=torch.float16)
        m_i = hl.full([tile_b, tile_m, 1], float("-inf"), dtype=torch.float16)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float16)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            k = k_view[tile_b, :, tile_n]
            qk = torch.bmm(q, k)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1, keepdim=True) * qk_scale_dev)
            m_ij_broad = broadcast(m_ij, 2, [m_ij.size(0), m_ij.size(1), tile_n])
            qk = qk * qk_scale_dev - m_ij_broad
            p = torch.exp(qk)
            l_ij = torch.sum(p, -1, keepdim=True)
            alpha = torch.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        m_i += torch.log(l_i)
        l_i_broadcast = broadcast(l_i, 2, [l_i.size(0), l_i.size(1), head_dim])
        acc = acc / l_i_broadcast
        out_[tile_b, tile_m, :] = acc.to(out_.dtype)
    return out_.view(q_in.size())


class FlashAttention(LoomKernel):

    kernel_name = "Flash Attention"

    B: int = 256
    L: int = 4096
    d: int = 128

    assume_divisible_tiles = True

    # Assign the helion-decorated function as a class attribute.
    # We cannot stack @staticmethod with @helion.kernel because the helion
    # decorator returns a custom object, not a plain callable.
    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_flash__attention)

    @classmethod
    def bind_args(cls) -> tuple:
        """Return concrete input tensors that define B, L, d at MLIR-gen time."""
        q = torch.randn([cls.B, cls.L, cls.d], dtype=torch.float16)
        k = torch.randn([cls.B, cls.L, cls.d], dtype=torch.float16)
        v = torch.randn([cls.B, cls.L, cls.d], dtype=torch.float16)
        return (q, k, v)


if __name__ == "__main__":
    FlashAttention.run()
