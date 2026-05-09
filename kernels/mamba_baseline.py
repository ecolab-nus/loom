def prepare_mamba2_chunk_scan_x(x: torch.Tensor) -> torch.Tensor:
    """
    Match the layout Helion uses after x.transpose(1, 2).

    Input:
        x: (batch, seqlen, nheads, headdim)

    Return:
        x_t: (batch, nheads, seqlen, headdim)
    """

    return x.transpose(1, 2).contiguous()


def prepare_mamba2_chunk_scan_C(C: torch.Tensor) -> torch.Tensor:
    """
    Match the layout Helion uses after C.transpose(1, 2).

    Input:
        C: (batch, seqlen, ngroups, dstate)

    Return:
        C_t: (batch, ngroups, seqlen, dstate)
    """

    return C.transpose(1, 2).contiguous()


def prepare_mamba2_chunk_scan_prev_states(prev_states: torch.Tensor) -> torch.Tensor:
    """
    Match the layout Helion uses before entering the tiled loop.

    Input:
        prev_states: (batch, nchunks, nheads, headdim, dstate)

    Return:
        prev_states_t: (batch, nchunks, nheads, dstate, headdim)

    Do this before converting to the TTNN device tensor so the TTNN compute path
    does not perform a layout-changing transpose/permute internally.
    """

    return prev_states.transpose(3, 4).contiguous()


def prepare_mamba2_chunk_scan_inputs(
    cb: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    dA: torch.Tensor,
    C: torch.Tensor,
    prev_states: torch.Tensor,
    D: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Prepare host tensors for the TTNN compute-only kernel."""

    return (
        cb,
        prepare_mamba2_chunk_scan_x(x),
        dt,
        dA,
        prepare_mamba2_chunk_scan_C(C),
        prepare_mamba2_chunk_scan_prev_states(prev_states),
        D,
    )


def ttnn_mamba2_chunk_scan_compute_coarse_causal(
    cb_tt,
    x_tt,
    dt_tt,
    dA_tt,
    C_tt,
    prev_tt,
    D_host: torch.Tensor,
    *,
    batch: int,
    nchunks: int,
    ngroups: int,
    chunk_size: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    dstate: int,
    block_size: int,
    dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    """
    TTNN compute-only Mamba2 chunk scan with coarse-grained causal mask.

    Coarse causal rule:
        m-block i only reads k-blocks [0, ..., i].

    Assumptions:
        - chunk_size % block_size == 0
        - m block size == block_size
        - k block size == 64, matching the Helion block_k registration
        - no fine-grained causal predicate m >= k
        - inputs are already TTNN device tensors

    Shapes:
        cb_tt:   (batch, nchunks, ngroups, chunk_size, chunk_size)
        x_tt:    (batch, nheads, seqlen, headdim)
                 Use prepare_mamba2_chunk_scan_x(x) before converting the
                 tensor to TTNN.
        dt_tt:   (batch, nheads, nchunks, chunk_size)
        dA_tt:   (batch, nheads, nchunks, chunk_size)
        C_tt:    (batch, ngroups, seqlen, dstate)
                 Use prepare_mamba2_chunk_scan_C(C) before converting the
                 tensor to TTNN.
        prev_tt: (batch, nchunks, nheads, dstate, headdim)
                 Use prepare_mamba2_chunk_scan_prev_states(prev_states) before
                 converting the tensor to TTNN.

    Returns:
        outputs: list of TTNN tensors.
        Each tensor has shape (1, 1, chunk_size, headdim), representing one
        output chunk in Helion's transposed output layout
        (batch, nheads, seqlen, headdim).
    """

    assert nheads % ngroups == 0
    assert seqlen % chunk_size == 0
    assert chunk_size % block_size == 0
    block_k = 64
    assert chunk_size % block_k == 0
    assert block_size % block_k == 0

    heads_per_group = nheads // ngroups
    outputs = []

    for b in range(batch):
        for h in range(nheads):
            g = h // heads_per_group

            for c in range(nchunks):
                seq_start = c * chunk_size

                # ------------------------------------------------------------
                # 1. Previous-state contribution
                # ------------------------------------------------------------

                C_local = tt_slice(
                    C_tt,
                    [b, g, seq_start, 0],
                    [b + 1, g + 1, seq_start + chunk_size, dstate],
                )
                C_local = ttnn.reshape(
                    C_local,
                    (1, 1, chunk_size, dstate),
                )

                prev_local = tt_slice(
                    prev_tt,
                    [b, c, h, 0, 0],
                    [b + 1, c + 1, h + 1, dstate, headdim],
                )
                prev_local = ttnn.reshape(
                    prev_local,
                    (1, 1, dstate, headdim),
                )

                acc_prev = ttnn.matmul(
                    C_local,
                    prev_local,
                    memory_config=memory_config,
                    dtype=dtype,
                )

                dA_m_full = tt_slice(
                    dA_tt,
                    [b, h, c, 0],
                    [b + 1, h + 1, c + 1, chunk_size],
                )
                dA_m_full = ttnn.reshape(
                    dA_m_full,
                    (1, 1, chunk_size, 1),
                )

                scale_m = ttnn.exp(dA_m_full)
                acc_prev = ttnn.mul(acc_prev, scale_m)

                # ------------------------------------------------------------
                # 2. Intra-chunk scan contribution
                #
                # Coarse causal:
                #   for each block_size-wide m-block:
                #       accumulate 64-wide k-blocks from 0 to current m-block
                #
                # No fine-grained:
                #   pred = m >= k
                # ------------------------------------------------------------

                acc_scan_m_blocks = []

                for m0 in range(0, chunk_size, block_size):
                    m1 = m0 + block_size

                    acc_scan_m = None

                    for k0 in range(0, m1, block_k):
                        k1 = k0 + block_k

                        cb_mk = tt_slice(
                            cb_tt,
                            [b, c, g, m0, k0],
                            [b + 1, c + 1, g + 1, m1, k1],
                        )
                        cb_mk = ttnn.reshape(
                            cb_mk,
                            (1, 1, block_size, block_k),
                        )

                        dA_m = tt_slice(
                            dA_tt,
                            [b, h, c, m0],
                            [b + 1, h + 1, c + 1, m1],
                        )
                        dA_m = ttnn.reshape(
                            dA_m,
                            (1, 1, block_size, 1),
                        )

                        dA_k = tt_slice(
                            dA_tt,
                            [b, h, c, k0],
                            [b + 1, h + 1, c + 1, k1],
                        )
                        dA_k = ttnn.reshape(
                            dA_k,
                            (1, 1, 1, block_k),
                        )

                        dt_k = tt_slice(
                            dt_tt,
                            [b, h, c, k0],
                            [b + 1, h + 1, c + 1, k1],
                        )
                        dt_k = ttnn.reshape(
                            dt_k,
                            (1, 1, 1, block_k),
                        )

                        dA_diff = ttnn.sub(dA_m, dA_k)
                        exp_dA_diff = ttnn.exp(dA_diff)

                        scan_weight_mk = ttnn.mul(cb_mk, exp_dA_diff)
                        scan_weight_mk = ttnn.mul(scan_weight_mk, dt_k)

                        x_k = tt_slice(
                            x_tt,
                            [b, h, seq_start + k0, 0],
                            [b + 1, h + 1, seq_start + k1, headdim],
                        )
                        x_k = ttnn.reshape(
                            x_k,
                            (1, 1, block_k, headdim),
                        )

                        partial = ttnn.matmul(
                            scan_weight_mk,
                            x_k,
                            memory_config=memory_config,
                            dtype=dtype,
                        )

                        if acc_scan_m is None:
                            acc_scan_m = partial
                        else:
                            acc_scan_m = ttnn.add(acc_scan_m, partial)

                    acc_scan_m_blocks.append(acc_scan_m)

                acc_scan = ttnn.concat(acc_scan_m_blocks, dim=2)

                # ------------------------------------------------------------
                # 3. D residual
                # ------------------------------------------------------------

                x_local = tt_slice(
                    x_tt,
                    [b, h, seq_start, 0],
                    [b + 1, h + 1, seq_start + chunk_size, headdim],
                )
                x_local = ttnn.reshape(
                    x_local,
                    (1, 1, chunk_size, headdim),
                )

                D_h = float(D_host[h].item())
                x_residual = ttnn.mul(x_local, D_h)

                acc = ttnn.add(acc_prev, acc_scan)
                acc = ttnn.add(acc, x_residual)

                outputs.append(acc)

    return outputs


def ttnn_mamba2_chunk_scan_compute(
    cb_tt,
    x_tt,
    dt_tt,
    dA_tt,
    C_tt,
    prev_tt,
    D_host: torch.Tensor,
    *,
    batch: int,
    nchunks: int,
    ngroups: int,
    chunk_size: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    dstate: int,
):
    """
    Compute-only TTNN implementation.

    Inputs are already TTNN device tensors.

    cb_tt:   (batch, nchunks, ngroups, chunk_size, chunk_size)
    x_tt:    (batch, nheads, seqlen, headdim)
             Use prepare_mamba2_chunk_scan_x(x) before converting the tensor
             to TTNN.
    dt_tt:   (batch, nheads, nchunks, chunk_size)
    dA_tt:   (batch, nheads, nchunks, chunk_size)
    C_tt:    (batch, ngroups, seqlen, dstate)
             Use prepare_mamba2_chunk_scan_C(C) before converting the tensor
             to TTNN.
    prev_tt: (batch, nchunks, nheads, dstate, headdim)
             Use prepare_mamba2_chunk_scan_prev_states(prev_states) before
             converting the tensor to TTNN.

    Returns:
        outputs: list of TTNN tensors.
        Each tensor has shape (1, 1, chunk_size, headdim), representing one
        output chunk in Helion's transposed output layout
        (batch, nheads, seqlen, headdim).

    Note:
        This function intentionally does not call:
            - ttnn.from_torch
            - ttnn.to_torch
            - ttnn.open_device
            - ttnn.close_device

        Therefore, it is suitable for performance measurement.
    """

    assert nheads % ngroups == 0
    assert seqlen % chunk_size == 0

    heads_per_group = nheads // ngroups
    outputs = []

    for b in range(batch):
        for h in range(nheads):
            g = h // heads_per_group

            for c in range(nchunks):
                seq_start = c * chunk_size

                # ------------------------------------------------------------
                # 1. Previous-state contribution
                #
                # acc_prev[m, p] =
                #   sum_s C_T[b, g, seq_start + m, s]
                #       * prev_states_T[b, c, h, s, p]
                # ------------------------------------------------------------

                C_local = tt_slice(
                    C_tt,
                    [b, g, seq_start, 0],
                    [b + 1, g + 1, seq_start + chunk_size, dstate],
                )
                C_local = ttnn.reshape(
                    C_local,
                    (1, 1, chunk_size, dstate),
                )

                prev_local = tt_slice(
                    prev_tt,
                    [b, c, h, 0, 0],
                    [b + 1, c + 1, h + 1, dstate, headdim],
                )
                prev_local = ttnn.reshape(
                    prev_local,
                    (1, 1, dstate, headdim),
                )

                acc_prev = ttnn.matmul(
                    C_local,
                    prev_local,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=ttnn.bfloat16,
                )

                # dA_m: (1, 1, chunk_size, 1)
                dA_m = tt_slice(
                    dA_tt,
                    [b, h, c, 0],
                    [b + 1, h + 1, c + 1, chunk_size],
                )
                dA_m = ttnn.reshape(
                    dA_m,
                    (1, 1, chunk_size, 1),
                )

                # Helion code:
                #   exp2(dA * 1.44269504)
                #
                # This is equivalent to:
                #   exp(dA)
                scale_m = ttnn.exp(dA_m)
                acc_prev = ttnn.mul(acc_prev, scale_m)

                # ------------------------------------------------------------
                # 2. Intra-chunk scan contribution
                #
                # scan_weight[m, k] =
                #   cb[m, k] * exp(dA[m] - dA[k]) * dt[k]
                #
                # acc_scan[m, p] =
                #   sum_k scan_weight[m, k] * x[k, p]
                # ------------------------------------------------------------

                cb_local = tt_slice(
                    cb_tt,
                    [b, c, g, 0, 0],
                    [b + 1, c + 1, g + 1, chunk_size, chunk_size],
                )
                cb_local = ttnn.reshape(
                    cb_local,
                    (1, 1, chunk_size, chunk_size),
                )

                # dA_k: (1, 1, 1, chunk_size)
                dA_k = tt_slice(
                    dA_tt,
                    [b, h, c, 0],
                    [b + 1, h + 1, c + 1, chunk_size],
                )
                dA_k = ttnn.reshape(
                    dA_k,
                    (1, 1, 1, chunk_size),
                )

                # dt_k: (1, 1, 1, chunk_size)
                dt_k = tt_slice(
                    dt_tt,
                    [b, h, c, 0],
                    [b + 1, h + 1, c + 1, chunk_size],
                )
                dt_k = ttnn.reshape(
                    dt_k,
                    (1, 1, 1, chunk_size),
                )

                dA_diff = ttnn.sub(dA_m, dA_k)
                exp_dA_diff = ttnn.exp(dA_diff)

                scan_weight = ttnn.mul(cb_local, exp_dA_diff)
                scan_weight = ttnn.mul(scan_weight, dt_k)

                x_local = tt_slice(
                    x_tt,
                    [b, h, seq_start, 0],
                    [b + 1, h + 1, seq_start + chunk_size, headdim],
                )
                x_local = ttnn.reshape(
                    x_local,
                    (1, 1, chunk_size, headdim),
                )

                acc_scan = ttnn.matmul(
                    scan_weight,
                    x_local,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=ttnn.bfloat16,
                )

                # ------------------------------------------------------------
                # 3. D residual
                #
                # acc += x[m, p] * D[h]
                # ------------------------------------------------------------

                D_h = float(D_host[h].item())
                x_residual = ttnn.mul(x_local, D_h)

                acc = ttnn.add(acc_prev, acc_scan)
                acc = ttnn.add(acc, x_residual)

                outputs.append(acc)

    return outputs
