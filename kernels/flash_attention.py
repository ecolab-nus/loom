"""Flash Attention kernel for the Loom pipeline.

Standalone CLI script. Run from the repo root:

    uv run python kernels/flash_attention.py --config kernels/config_files/flash_attention.json --njobs 8 --debug --topk-candidates 1 --topk-block-size 1

This script inherits the full Loom CLI and pipeline from LoomKernel.
To write your own kernel, copy this file, replace the kernel body
and bind_args tensors, and keep the __main__ block unchanged.
"""

from __future__ import annotations

import math
import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args
from helion_mlir.custom_op import broadcast, set_memory_space


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
            k = set_memory_space(k_view[tile_b, :, tile_n], local_mem_kind=1)
            qk = torch.bmm(q, k)
            qk = qk * qk_scale_dev
            m_ij = torch.maximum(m_i, torch.amax(qk, -1, keepdim=True))
            m_ij_broad = broadcast(m_ij, 2, [m_ij.size(0), m_ij.size(1), tile_n])
            qk = qk - m_ij_broad
            p = torch.exp(qk)
            alpha = torch.exp(m_i - m_ij)
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = acc * alpha
            acc = acc + torch.bmm(p, v)
            l_ij = torch.sum(p, -1, keepdim=True)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
        m_i += torch.log(l_i)
        l_i_broadcast = broadcast(l_i, 2, [l_i.size(0), l_i.size(1), head_dim])
        acc = acc / l_i_broadcast
        out_[tile_b, tile_m, :] = acc.to(out_.dtype)
    return out_.view(q_in.size())


class FlashAttention(LoomKernel):

    kernel_name = "Flash Attention"

    B: int = 2
    L: int = 4096
    H: int = 128
    d: int = int(1024 * 64 / H)
    _logical_B: int = B

    assume_divisible = True

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

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        cls = type(self)
        logical_b = cls._logical_B
        if shape:
            for key, value in shape.items():
                if key == "B":
                    logical_b = value
                else:
                    setattr(cls, key, value)

        cls._logical_B = logical_b
        cls.d = int(1024 * 64 / cls.H)
        cls.B = logical_b * cls.H

    @classmethod
    def bind_args(cls) -> tuple:
        """Return concrete input tensors that define B*H, L, d at MLIR-gen time."""
        q = torch.empty([cls.B, cls.L, cls.d], dtype=torch.float16)
        k = torch.empty([cls.B, cls.L, cls.d], dtype=torch.float16)
        v = torch.empty([cls.B, cls.L, cls.d], dtype=torch.float16)
        return (q, k, v)


if __name__ == "__main__":
    try:
        shape, normalized_argv = resolve_kernel_shape_args(sys.argv)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    sys.argv = normalized_argv
    kernel = FlashAttention(shape)
    kernel.run()
