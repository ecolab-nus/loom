"""Mamba chunk-scan kernel for the Loom pipeline.

Standalone CLI script. Run from the repo root:

    python kernels/mamba_chunk_scan_wh/L4096_N128_H128_G8_D128_C512.py \
        --config kernels/config_files/mamba_chunk_scan_wh/L4096_N128_H128_G8_D128_C512.json \
        --njobs 1 \
        --debug

This script inherits the full Loom CLI and pipeline from LoomKernel.
To write your own kernel, copy this file, replace the kernel body
and bind_args tensors, and keep the __main__ block unchanged.
"""

from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args

from helion_mlir.custome_op import broadcast  # registers the op with Helion's decorator API


def _mamba_chunk_scan(
    cb: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    C: torch.Tensor,
    prev_states: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """
    Argument:
        cb: (batch, nchunks, ngroups, chunk_size, chunk_size)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        C: (batch, seqlen, ngroups, dstate)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        D: (nheads,)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """

    batch, nchunks, ngroups, chunk_size, _ = cb.shape
    _, seqlen, nheads, headdim = x.shape
    _, _, _, dstate = C.shape
    assert nchunks == (seqlen + chunk_size - 1) // chunk_size

    block_m = hl.register_block_size(chunk_size)
    block_n = hl.register_block_size(headdim)
    block_k = hl.register_block_size(64, 64)

    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert prev_states.shape == (batch, nchunks, nheads, headdim, dstate)
    assert D.shape == (nheads,)

    dtype = cb.dtype
    accum_dtype = torch.float16
    assert (
        x.dtype
        == dt.dtype
        == dA_cumsum.dtype
        == C.dtype
        == prev_states.dtype
        == D.dtype
        == dtype
    )
    prev_states_T = prev_states.transpose(3, 4)

    out_ = torch.empty_like(x)

    for tile_h, tile_m, tile_n, tile_b, tile_c in hl.tile(
        [nheads, chunk_size, headdim, batch, nchunks],
        block_size=[1, block_m, block_n, 1, 1],
    ):
        # tile_h: head tile (size 1)
        # tile_m: chunk-local sequence rows (M axis)
        # tile_n: head-dim columns (N axis)
        # tile_b: batch tile (size 1)
        # tile_c: chunk id tile (size 1)
        acc_o = hl.zeros([tile_m, tile_n], dtype=accum_dtype)

        # dA_cumsum_local_m: [tile_m]
        dA_cumsum_local_m = dA_cumsum[tile_b.begin, tile_h.begin, tile_c.begin, tile_m]
        dA_cumsum_local_m_bc_n = broadcast(
            dA_cumsum_local_m,
            1,
            [dA_cumsum_local_m.size(0), tile_n],
        )

        # scale_m_local: [tile_m, tile_n]
        scale_m_local = torch.exp(dA_cumsum_local_m_bc_n)

        # C_local: [tile_m, dstate]
        # row index = tile_c * chunk_size + tile_m.index
        C_local = C[
            tile_b.begin,
            tile_m.index + tile_c.begin * chunk_size,
            tile_h.begin // (nheads // ngroups),
            :,
        ]
        # prev_states_local: [dstate, tile_b]
        prev_states_local = prev_states_T[
            tile_b.begin, tile_c.begin, tile_h.begin, :, tile_n
        ]
        # hl.dot([tile_m, dstate], [dstate, tile_n]) -> [tile_m, tile_n]
        acc_o = hl.dot(C_local, prev_states_local, acc=acc_o)
        acc_o *= scale_m_local

        for tile_k in hl.tile((tile_m.id + 1) * block_m, block_size=block_k):
            # cb_local: [tile_m, tile_k]
            cb_local = cb[
                tile_b.begin,
                tile_c.begin,
                tile_h.begin // (nheads // ngroups),
                tile_m,
                tile_k,
            ]
            # dA_cumsum_local_k: [tile_k]
            dA_cumsum_local_k = dA_cumsum[
                tile_b.begin, tile_h.begin, tile_c.begin, tile_k
            ]
            dA_cumsum_local_m_bc_k = broadcast(
                dA_cumsum_local_m,
                1,
                [dA_cumsum_local_m.size(0), tile_k],
            )
            dA_cumsum_local_k = broadcast(
                dA_cumsum_local_k,
                0,
                [tile_m, dA_cumsum_local_k.size(0)],
            )
            # broadcast to [tile_m, tile_k]
            cb_local *= torch.exp(dA_cumsum_local_m_bc_k - dA_cumsum_local_k)
            # dt_local: [tile_k]
            dt_local = dt[tile_b.begin, tile_h.begin, tile_c.begin, tile_k]
            # dt_local[None, :]: [1, tile_k], broadcast over tile_m axis
            dt_local = broadcast(dt_local, 0, [tile_m, dt_local.size(0)])
            cb_local *= dt_local

            # x_local: [tile_k, tile_n]
            x_local = x[
                tile_b.begin,
                tile_c.begin * chunk_size + tile_k.index,
                tile_h.begin,
                tile_n,
            ]
            # hl.dot([tile_m, tile_k], [tile_k, tile_n]) -> [tile_m, tile_n]
            acc_o = hl.dot(cb_local, x_local, acc=acc_o)

        # D_local: scalar
        D_local = D[tile_h.begin]
        # x_residual: [tile_m, tile_n]
        x_residual = x[
            tile_b.begin,
            tile_c.begin * chunk_size + tile_m.index,
            tile_h.begin,
            tile_n,
        ]
        # D_local scalar broadcasts to [tile_m, tile_n]
        acc_o += x_residual * D_local

        # out[...] tile: [tile_m, tile_n]
        out_[
            tile_b.begin,
            tile_c.begin * chunk_size + tile_m.index,
            tile_h.begin,
            tile_n,
        ] = acc_o.to(dtype=dtype)

    return out_


class MambaChunkScan(LoomKernel):

    kernel_name = "mamba_chunk_scan"

    BATCH: int = 2
    SEQLEN: int = 4096
    NHEADS: int = 128
    HEADDIM: int = 128
    NGROUPS: int = 8
    DSTATE: int = 128
    CHUNK_SIZE: int = 512

    assume_divisible_tiles: bool = True    

    kernel = helion.kernel(
        static_shapes=False,
    )(_mamba_chunk_scan)

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            cls = type(self)
            key_to_attr = {
                "B": "BATCH",
                "L": "SEQLEN",
                "N": "NHEADS",
                "H": "HEADDIM",
                "G": "NGROUPS",
                "D": "DSTATE",
                "C": "CHUNK_SIZE",
            }
            for key, value in shape.items():
                attr = key_to_attr.get(key, key)
                setattr(cls, attr, value)

    @classmethod
    def bind_args(cls) -> tuple:
        """Return concrete input tensors that define kernel shapes at MLIR-gen time."""
        nchunks = (cls.SEQLEN + cls.CHUNK_SIZE - 1) // cls.CHUNK_SIZE

        cb = torch.randn(
            [cls.BATCH, nchunks, cls.NGROUPS, cls.CHUNK_SIZE, cls.CHUNK_SIZE],
            dtype=torch.float16,
        )
        x = torch.randn([cls.BATCH, cls.SEQLEN, cls.NHEADS, cls.HEADDIM], dtype=torch.float16)
        dt = torch.randn([cls.BATCH, cls.NHEADS, nchunks, cls.CHUNK_SIZE], dtype=torch.float16)
        dA_cumsum = torch.randn([cls.BATCH, cls.NHEADS, nchunks, cls.CHUNK_SIZE], dtype=torch.float16)
        C = torch.randn([cls.BATCH, cls.SEQLEN, cls.NGROUPS, cls.DSTATE], dtype=torch.float16)
        prev_states = torch.randn(
            [cls.BATCH, nchunks, cls.NHEADS, cls.HEADDIM, cls.DSTATE],
            dtype=torch.float16,
        )
        D = torch.randn([cls.NHEADS], dtype=torch.float16)

        return (cb, x, dt, dA_cumsum, C, prev_states, D)


if __name__ == "__main__":
    try:
        shape, normalized_argv = resolve_kernel_shape_args(sys.argv)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    sys.argv = normalized_argv
    kernel = MambaChunkScan(shape)
    kernel.run()
