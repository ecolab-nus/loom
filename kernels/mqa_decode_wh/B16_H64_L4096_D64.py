"""MQA decode kernel for the Loom pipeline.

Standalone CLI script. Run from the repo root:

    python kernels/mqa_decode_wh/B16_H64_L4096_D64.py --config kernels/config_files/mqa_decode_wh/B16_H64_L4096_D64.json --njobs 1 --debug

This script inherits the full Loom CLI and pipeline from LoomKernel.
To write your own kernel, copy this file, replace the kernel body
and bind_args tensors, and keep the __main__ block unchanged.
"""

import math

import torch
import helion
import helion.language as hl

from loom import LoomKernel

from helion_mlir.custome_op import gather, broadcast  # registers the op with Helion's decorator API


def _mqa_decode(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    batch, _, num_q_head, head_dim = q_in.size()
    kvseq_len = k_in.size(-2)

    num_q_head = hl.specialize(num_q_head)
    head_dim = hl.specialize(head_dim)

    q_view = q_in.reshape([batch, num_q_head, head_dim])
    k_view = k_in.reshape([batch, kvseq_len, head_dim]).transpose(1, 2)
    v_view = v_in.reshape([batch, kvseq_len, head_dim])

    sm_scale = 1.0 / math.sqrt(head_dim)
    out = torch.zeros([batch, num_q_head, head_dim], dtype=torch.float16)

    for tile_b, tile_s in hl.tile([batch, kvseq_len]):
        qk_scale = hl.full([], sm_scale, dtype=torch.float16)

        m_i = hl.full([tile_b, num_q_head, 1], float("-inf"), dtype=torch.float16)
        l_i = hl.full([tile_b, num_q_head, 1], 1.0, dtype=torch.float16)
        acc = hl.zeros([tile_b, num_q_head, head_dim], dtype=torch.float16)
        q = q_view[tile_b, :, :]

        for tile_n in hl.tile(tile_s.begin, tile_s.end):
            k = k_view[tile_b, :, tile_n]
            qk = torch.bmm(q, k)

            m_qk = torch.amax(qk, -1, keepdim=True)
            m_ij = torch.maximum(m_i, m_qk * qk_scale)
            m_ij_broadcast = broadcast(m_ij, 2, [m_ij.size(0), m_ij.size(1), tile_n])
            qk = qk * qk_scale - m_ij_broadcast

            p = torch.exp(qk)
            l_ij = torch.sum(p, -1, keepdim=True)
            alpha = torch.exp(m_i - m_ij)

            l_i = l_i * alpha + l_ij
            alpha_broadcast = broadcast(alpha, 2, [alpha.size(0), alpha.size(1), head_dim])
            acc = acc * alpha_broadcast

            v = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij

        split_lse = torch.log(l_i) + m_i
        l_i_bc = broadcast(l_i, 2, [l_i.size(0), l_i.size(1), head_dim])
        acc = acc / l_i_bc

        gathered_lse = gather(tile_s, split_lse)
        gathered_acc = gather(tile_s, acc)

        if tile_s.id == 0:
            max_lse = torch.amax(gathered_lse, 0)
            weights = torch.exp(gathered_lse - max_lse)
            lse_sum = torch.sum(weights, 0)
            norm_scale = weights / lse_sum
            norm_scale = broadcast(
                norm_scale,
                3,
                [norm_scale.size(0), norm_scale.size(1), norm_scale.size(2), head_dim],
            )
            weighted_acc = torch.sum(gathered_acc * norm_scale, 0)
            out[tile_b, :, :] = weighted_acc

    return out.view(q_in.size())


class MQADecode(LoomKernel):

    kernel_name = "mqa_decode"

    B: int = 16
    H: int = 64
    L: int = 4096
    D: int = 64

    assume_divisible_tiles: bool = True

    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_mqa_decode)

    @classmethod
    def bind_args(cls) -> tuple:
        """Return concrete input tensors that define B, H, L, D at MLIR-gen time."""
        q = torch.randn([cls.B, 1, cls.H, cls.D], dtype=torch.float16)
        k = torch.randn([cls.B, 1, cls.L, cls.D], dtype=torch.float16)
        v = torch.randn([cls.B, 1, cls.L, cls.D], dtype=torch.float16)
        return (q, k, v)


if __name__ == "__main__":
    MQADecode.run()
