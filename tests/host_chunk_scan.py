import argparse
import inspect
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path


def _bootstrap_tt_metal_home():
    if os.environ.get("TT_METAL_HOME"):
        return

    for parent in Path(__file__).resolve().parents:
        if (parent / "runtime" / "hw").is_dir() and (parent / "runtime" / "sfpi").is_dir():
            os.environ["TT_METAL_HOME"] = str(parent)
            return


_bootstrap_tt_metal_home()

import torch
import ttnn
from models.common.utility_functions import profiler


KERNELS_DIR = Path(__file__).resolve().parent / "kernels"
sys.path.insert(0, str(KERNELS_DIR))

import host_ttnn as generated_host_ttnn

run_generated_chunk_scan = generated_host_ttnn.run


def _log(message):
    print(message, flush=True)


@dataclass(frozen=True)
class ChunkScanConfig:
    batch: int
    seqlen: int
    nheads: int
    headdim: int
    ngroups: int
    dstate: int
    chunk_size: int
    block_size: int

    @property
    def nchunks(self):
        return self.seqlen // self.chunk_size


@dataclass
class ChunkScanHostInputs:
    cb: torch.Tensor
    x: torch.Tensor
    dt: torch.Tensor
    dA: torch.Tensor
    C: torch.Tensor
    prev: torch.Tensor
    D: torch.Tensor


@dataclass
class ChunkScanDeviceTensors:
    cb: ttnn.Tensor
    x: ttnn.Tensor
    dt: ttnn.Tensor
    dA: ttnn.Tensor
    C: ttnn.Tensor
    prev: ttnn.Tensor
    D: ttnn.Tensor
    D_host: torch.Tensor
    dst: ttnn.Tensor | None = None


GENERATED_CONFIG = ChunkScanConfig(
    batch=2,
    seqlen=4096,
    nheads=64,
    headdim=128,
    ngroups=4,
    dstate=128,
    chunk_size=256,
    block_size=64,
)


SHAPE_SPEC_RE = re.compile(
    r"(?:^|[^A-Za-z0-9])"
    r"L(?P<seq_len>\d+)_N(?P<nhead>\d+)_H(?P<head_dim>\d+)_"
    r"G(?P<ngroups>\d+)_D(?P<dstate>\d+)_C(?P<chunk_size>\d+)"
    r"(?=$|[^A-Za-z0-9])",
    re.IGNORECASE,
)


SHAPE_ARG_DEFAULTS = {
    "seq_len": GENERATED_CONFIG.seqlen,
    "nhead": GENERATED_CONFIG.nheads,
    "head_dim": GENERATED_CONFIG.headdim,
    "ngroups": GENERATED_CONFIG.ngroups,
    "dstate": GENERATED_CONFIG.dstate,
    "chunk_size": GENERATED_CONFIG.chunk_size,
}


def _parse_shape_spec(shape_spec):
    match = SHAPE_SPEC_RE.search(shape_spec)
    if match is None:
        raise ValueError(
            f"invalid shape spec '{shape_spec}', expected something like L1024_N32_H128_G4_D128_C128"
        )
    return {name: int(value) for name, value in match.groupdict().items()}


def _apply_shape_spec_args(args, parser):
    if args.shape and args.shape_spec and args.shape != args.shape_spec:
        parser.error("use either positional SHAPE or --shape, not both")

    shape_spec = args.shape_spec or args.shape
    shape_values = {}
    if shape_spec:
        try:
            shape_values = _parse_shape_spec(shape_spec)
        except ValueError as exc:
            parser.error(str(exc))

    for name, default_value in SHAPE_ARG_DEFAULTS.items():
        if getattr(args, name) is None:
            setattr(args, name, shape_values.get(name, default_value))

    return args


def _validate_config(cfg):
    if cfg.nheads % cfg.ngroups != 0:
        raise ValueError(f"nheads must be divisible by ngroups, got {cfg.nheads} and {cfg.ngroups}")
    if cfg.seqlen % cfg.chunk_size != 0:
        raise ValueError(f"seqlen must be divisible by chunk_size, got {cfg.seqlen} and {cfg.chunk_size}")
    if cfg.chunk_size % cfg.block_size != 0:
        raise ValueError(f"chunk_size must be divisible by block_size, got {cfg.chunk_size} and {cfg.block_size}")
    if cfg.chunk_size % 64 != 0:
        raise ValueError(f"chunk_size must be divisible by block_k=64, got {cfg.chunk_size}")
    if cfg.block_size % 64 != 0:
        raise ValueError(f"block_size must be divisible by block_k=64, got {cfg.block_size}")
    if cfg.headdim % 32 != 0 or cfg.dstate % 32 != 0:
        raise ValueError(f"headdim and dstate must be tile aligned, got {cfg.headdim} and {cfg.dstate}")


def _torch_rand_bf16(shape, seed, low, high):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tensor = torch.rand(shape, generator=generator, dtype=torch.float32)
    tensor = tensor * (high - low) + low
    return tensor.to(torch.bfloat16)


def prepare_mamba2_chunk_scan_x(x):
    return x.permute(0, 2, 1, 3).contiguous()


def prepare_mamba2_chunk_scan_C(C):
    return C.permute(0, 2, 1, 3).contiguous()


def prepare_mamba2_chunk_scan_prev_states(prev_states):
    return prev_states.transpose(3, 4).contiguous()


def prepare_mamba2_chunk_scan_inputs(cb, x, dt, dA, C, prev_states, D):
    return ChunkScanHostInputs(
        cb=cb.contiguous(),
        x=prepare_mamba2_chunk_scan_x(x),
        dt=dt.contiguous(),
        dA=dA.contiguous(),
        C=prepare_mamba2_chunk_scan_C(C),
        prev=prepare_mamba2_chunk_scan_prev_states(prev_states),
        D=D.contiguous(),
    )


def _build_inputs(cfg, low, high, tril_cb, raw_dt):
    cb = _torch_rand_bf16(
        (cfg.batch, cfg.nchunks, cfg.ngroups, cfg.chunk_size, cfg.chunk_size),
        seed=11,
        low=low,
        high=high,
    )
    if tril_cb:
        cb = torch.tril(cb)

    x = _torch_rand_bf16((cfg.batch, cfg.seqlen, cfg.nheads, cfg.headdim), seed=12, low=low, high=high)

    dt = _torch_rand_bf16((cfg.batch, cfg.nheads, cfg.nchunks, cfg.chunk_size), seed=13, low=low, high=high)
    if not raw_dt:
        dt = torch.nn.functional.softplus(dt.float()).to(torch.bfloat16)

    dA = _torch_rand_bf16((cfg.batch, cfg.nheads, cfg.nchunks, cfg.chunk_size), seed=14, low=low, high=high)
    C = _torch_rand_bf16((cfg.batch, cfg.seqlen, cfg.ngroups, cfg.dstate), seed=15, low=low, high=high)
    prev_states = _torch_rand_bf16(
        (cfg.batch, cfg.nchunks, cfg.nheads, cfg.headdim, cfg.dstate),
        seed=17,
        low=low,
        high=high,
    )
    D = _torch_rand_bf16((cfg.nheads,), seed=16, low=low, high=high)

    return prepare_mamba2_chunk_scan_inputs(cb, x, dt, dA, C, prev_states, D)


def _to_device_tile(tensor, device):
    return ttnn.as_tensor(
        tensor.contiguous(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_device_raw_pages(tensor, device):
    flat = tensor.contiguous().flatten()
    if flat.numel() % 1024 != 0:
        raise ValueError(f"raw page tensor must contain a whole number of tiles, got {flat.numel()} elements")
    return ttnn.as_tensor(
        flat.reshape(1, 1, -1, 1024),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _tilize_nfaces(row_major, rows, cols):
    if rows % 32 != 0 or cols % 32 != 0:
        raise ValueError(f"tilize_nfaces expects rows/cols divisible by 32, got {rows}x{cols}")
    flat = row_major.contiguous().flatten()
    batch = flat.numel() // (rows * cols)
    if batch * rows * cols != flat.numel():
        raise ValueError(f"input with {flat.numel()} elements is not divisible by rows*cols={rows * cols}")

    return (
        flat.reshape(batch, rows // 32, 2, 16, cols // 32, 2, 16)
        .permute(0, 1, 4, 2, 5, 3, 6)
        .contiguous()
        .flatten()
    )


def _untilize_nfaces(tiled, rows, cols):
    if rows % 32 != 0 or cols % 32 != 0:
        raise ValueError(f"untilize_nfaces expects rows/cols divisible by 32, got {rows}x{cols}")
    flat = tiled.contiguous().flatten()
    batch = flat.numel() // (rows * cols)
    if batch * rows * cols != flat.numel():
        raise ValueError(f"input with {flat.numel()} elements is not divisible by rows*cols={rows * cols}")

    return (
        flat.reshape(batch, rows // 32, cols // 32, 2, 2, 16, 16)
        .permute(0, 1, 3, 5, 2, 4, 6)
        .contiguous()
        .reshape(batch, rows, cols)
    )


def _prepare_device_tensors(device, host_inputs, cfg):
    D_for_device = host_inputs.D.reshape(1, 1, 1, cfg.nheads)

    return ChunkScanDeviceTensors(
        cb=_to_device_tile(host_inputs.cb, device),
        x=_to_device_tile(host_inputs.x, device),
        dt=_to_device_tile(host_inputs.dt, device),
        dA=_to_device_tile(host_inputs.dA, device),
        C=_to_device_tile(host_inputs.C, device),
        prev=_to_device_tile(host_inputs.prev, device),
        D=_to_device_tile(D_for_device, device),
        D_host=host_inputs.D.cpu(),
    )


def _prepare_generated_raw_device_tensors(device, host_inputs, cfg, with_dst):
    # This mirrors host_chunk_scan.cpp: the generated kernels address flat 2-D
    # TILED_NFACES pages, while dt/dA are raw BF16 streams with 2048-byte pages.
    cb_tiled = _tilize_nfaces(
        host_inputs.cb,
        rows=cfg.batch * cfg.nchunks * cfg.ngroups * cfg.chunk_size,
        cols=cfg.chunk_size,
    )
    x_tiled = _tilize_nfaces(
        host_inputs.x,
        rows=cfg.batch * cfg.nheads * cfg.seqlen,
        cols=cfg.headdim,
    )
    C_tiled = _tilize_nfaces(
        host_inputs.C,
        rows=cfg.batch * cfg.ngroups * cfg.seqlen,
        cols=cfg.dstate,
    )
    prev_tiled = _tilize_nfaces(
        host_inputs.prev,
        rows=cfg.batch * cfg.nchunks * cfg.nheads * cfg.dstate,
        cols=cfg.headdim,
    )

    dst = None
    if with_dst:
        dst_numel = cfg.batch * cfg.nheads * cfg.seqlen * cfg.headdim
        # Make unwritten pages obvious in correctness checks instead of inheriting stale DRAM contents.
        dst_tiled = torch.zeros(dst_numel, dtype=torch.bfloat16)
        dst = _to_device_raw_pages(dst_tiled, device)

    return ChunkScanDeviceTensors(
        cb=_to_device_raw_pages(cb_tiled, device),
        x=_to_device_raw_pages(x_tiled, device),
        dt=_to_device_raw_pages(host_inputs.dt, device),
        dA=_to_device_raw_pages(host_inputs.dA, device),
        C=_to_device_raw_pages(C_tiled, device),
        prev=_to_device_raw_pages(prev_tiled, device),
        D=None,
        D_host=host_inputs.D.cpu(),
        dst=dst,
    )


def tt_slice(x, start, end, step=None):
    if step is None:
        step = [1] * len(start)

    if len(start) != 4:
        return ttnn.slice(
            x,
            start,
            end,
            step,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return ttnn.slice(
        x,
        start,
        end,
        step,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        pad_value=0.0,
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

                C_local = tt_slice(
                    C_tt,
                    [b, g, seq_start, 0],
                    [b + 1, g + 1, seq_start + chunk_size, dstate],
                )
                C_local = ttnn.reshape(C_local, (1, 1, chunk_size, dstate))

                prev_local = tt_slice(
                    prev_tt,
                    [b, c, h, 0, 0],
                    [b + 1, c + 1, h + 1, dstate, headdim],
                )
                prev_local = ttnn.reshape(prev_local, (1, 1, dstate, headdim))

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
                dA_m_full = ttnn.reshape(dA_m_full, (1, 1, chunk_size, 1))

                scale_m = ttnn.exp(dA_m_full)
                acc_prev = ttnn.mul(acc_prev, scale_m)

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
                        cb_mk = ttnn.reshape(cb_mk, (1, 1, block_size, block_k))

                        dA_m = tt_slice(
                            dA_tt,
                            [b, h, c, m0],
                            [b + 1, h + 1, c + 1, m1],
                        )
                        dA_m = ttnn.reshape(dA_m, (1, 1, block_size, 1))

                        dA_k = tt_slice(
                            dA_tt,
                            [b, h, c, k0],
                            [b + 1, h + 1, c + 1, k1],
                        )
                        dA_k = ttnn.reshape(dA_k, (1, 1, 1, block_k))

                        dt_k = tt_slice(
                            dt_tt,
                            [b, h, c, k0],
                            [b + 1, h + 1, c + 1, k1],
                        )
                        dt_k = ttnn.reshape(dt_k, (1, 1, 1, block_k))

                        dA_diff = ttnn.sub(dA_m, dA_k)
                        exp_dA_diff = ttnn.exp(dA_diff)

                        scan_weight_mk = ttnn.mul(cb_mk, exp_dA_diff)
                        scan_weight_mk = ttnn.mul(scan_weight_mk, dt_k)

                        x_k = tt_slice(
                            x_tt,
                            [b, h, seq_start + k0, 0],
                            [b + 1, h + 1, seq_start + k1, headdim],
                        )
                        x_k = ttnn.reshape(x_k, (1, 1, block_k, headdim))

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

                x_local = tt_slice(
                    x_tt,
                    [b, h, seq_start, 0],
                    [b + 1, h + 1, seq_start + chunk_size, headdim],
                )
                x_local = ttnn.reshape(x_local, (1, 1, chunk_size, headdim))

                D_h = float(D_host[h].item())
                x_residual = ttnn.mul(x_local, D_h)

                acc = ttnn.add(acc_prev, acc_scan)
                acc = ttnn.add(acc, x_residual)

                outputs.append(acc)

    return outputs


def _assemble_official_outputs(outputs, cfg):
    out = torch.empty(
        cfg.batch,
        cfg.nheads,
        cfg.seqlen,
        cfg.headdim,
        dtype=torch.bfloat16,
    )

    idx = 0
    for b in range(cfg.batch):
        for h in range(cfg.nheads):
            for c in range(cfg.nchunks):
                seq_start = c * cfg.chunk_size
                seq_end = seq_start + cfg.chunk_size
                local = ttnn.to_torch(outputs[idx]).reshape(cfg.chunk_size, cfg.headdim)
                out[b, h, seq_start:seq_end, :] = local
                idx += 1

    return out.to(torch.float32)


def _launch_official_chunk_scan(device_tensors, cfg):
    return ttnn_mamba2_chunk_scan_compute_coarse_causal(
        device_tensors.cb,
        device_tensors.x,
        device_tensors.dt,
        device_tensors.dA,
        device_tensors.C,
        device_tensors.prev,
        device_tensors.D_host,
        batch=cfg.batch,
        nchunks=cfg.nchunks,
        ngroups=cfg.ngroups,
        chunk_size=cfg.chunk_size,
        seqlen=cfg.seqlen,
        nheads=cfg.nheads,
        headdim=cfg.headdim,
        dstate=cfg.dstate,
        block_size=cfg.block_size,
    )


def _run_official_chunk_scan(device, host_inputs, cfg):
    start = time.perf_counter()
    _log("  official: moving prepared inputs to device as TTNN TILE tensors")
    device_tensors = _prepare_device_tensors(device, host_inputs, cfg)
    _log(f"  official: device upload done in {time.perf_counter() - start:.3f}s")

    start = time.perf_counter()
    _log("  official: launching coarse causal TTNN reference")
    outputs = _launch_official_chunk_scan(device_tensors, cfg)
    ttnn.synchronize_device(device)
    launch_seconds = time.perf_counter() - start
    _log(f"  official: launch/sync done in {launch_seconds:.3f}s")

    start = time.perf_counter()
    _log("  official: reading and assembling output chunks")
    out = _assemble_official_outputs(outputs, cfg)
    _log(f"  official: output read/assemble done in {time.perf_counter() - start:.3f}s")
    return out


def _torch_chunk_scan_reference(host_inputs, cfg, output_bf16=False):
    cb = host_inputs.cb.float()
    x = host_inputs.x.float()
    dt = host_inputs.dt.float()
    dA = host_inputs.dA.float()
    C = host_inputs.C.float()
    prev = host_inputs.prev.float()
    D = host_inputs.D.float()

    out = torch.empty(cfg.batch, cfg.nheads, cfg.seqlen, cfg.headdim, dtype=torch.float32)
    heads_per_group = cfg.nheads // cfg.ngroups

    with torch.inference_mode():
        for b in range(cfg.batch):
            for h in range(cfg.nheads):
                g = h // heads_per_group
                for c in range(cfg.nchunks):
                    seq_start = c * cfg.chunk_size
                    seq_end = seq_start + cfg.chunk_size

                    C_local = C[b, g, seq_start:seq_end, :]
                    prev_local = prev[b, c, h, :, :]
                    acc = C_local @ prev_local

                    dA_chunk = dA[b, h, c, :]
                    acc = acc * torch.exp(dA_chunk)[:, None]

                    for m0 in range(0, cfg.chunk_size, cfg.block_size):
                        m1 = m0 + cfg.block_size
                        dA_m = dA_chunk[m0:m1]
                        dA_k = dA_chunk[:m1]
                        dt_k = dt[b, h, c, :m1]
                        scan_weight = cb[b, c, g, m0:m1, :m1]
                        scan_weight = scan_weight * torch.exp(dA_m[:, None] - dA_k[None, :])
                        scan_weight = scan_weight * dt_k[None, :]
                        x_k = x[b, h, seq_start:seq_start + m1, :]
                        acc[m0:m1, :] += scan_weight @ x_k

                    acc += x[b, h, seq_start:seq_end, :] * D[h]
                    out[b, h, seq_start:seq_end, :] = acc

    if output_bf16:
        return out.to(torch.bfloat16).to(torch.float32)
    return out


def _required_positional_count(func):
    signature = inspect.signature(func)
    params = [
        param
        for param in signature.parameters.values()
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        and param.default is param.empty
    ]
    has_varargs = any(param.kind == param.VAR_POSITIONAL for param in signature.parameters.values())
    return None if has_varargs else len(params)


def _parse_generated_order(order_text, required_count):
    if order_text != "auto":
        order = [item.strip() for item in order_text.split(",") if item.strip()]
        if required_count is not None and len(order) != required_count:
            raise ValueError(
                f"--generated_order has {len(order)} entries, but host_ttnn.run requires {required_count} positional args"
            )
        return order

    if required_count == 8:
        return ["cb", "x", "dt", "dA", "C", "D_host", "prev", "dst"]
    if required_count == 7:
        return ["cb", "x", "dt", "dA", "C", "prev", "D_host"]

    raise ValueError(
        f"{KERNELS_DIR / 'host_ttnn.py'} run currently requires {required_count} positional args. "
        "Chunk scan usually expects 8 args in order cb,x,dt,dA,C,D_host,prev,dst for generated host_ttnn, "
        "or 7 args in order cb,x,dt,dA,C,prev,D_host for an op-style wrapper. "
        "Regenerate kernels/host_ttnn.py for chunk scan or pass --generated_order explicitly."
    )


def _resolve_generated_order(order_text):
    return _parse_generated_order(order_text, _required_positional_count(run_generated_chunk_scan))


def _generated_arg_map(device_tensors):
    return {
        "cb": device_tensors.cb,
        "x": device_tensors.x,
        "dt": device_tensors.dt,
        "dA": device_tensors.dA,
        "da": device_tensors.dA,
        "C": device_tensors.C,
        "c": device_tensors.C,
        "prev": device_tensors.prev,
        "prev_states": device_tensors.prev,
        "D": device_tensors.D,
        "D_tt": device_tensors.D,
        "d": device_tensors.D,
        "D_host": device_tensors.D_host,
        "dst": device_tensors.dst,
        "out": device_tensors.dst,
    }


def _launch_generated_chunk_scan(device_tensors, generated_order):
    arg_map = _generated_arg_map(device_tensors)
    call_args = []
    for name in generated_order:
        if name not in arg_map:
            raise ValueError(
                f"Unknown generated argument name '{name}'. "
                f"Known names: {', '.join(sorted(arg_map.keys()))}"
            )
        value = arg_map[name]
        if value is None:
            raise ValueError(f"Generated argument '{name}' requires an allocated dst tensor")
        call_args.append(value)

    result = run_generated_chunk_scan(*call_args)
    if result is None:
        if device_tensors.dst is None:
            raise ValueError("host_ttnn.run returned None and no dst tensor was allocated")
        return device_tensors.dst
    if isinstance(result, (list, tuple)):
        if len(result) != 1:
            raise ValueError(f"host_ttnn.run returned {len(result)} outputs; expected one")
        return result[0]
    return result


def _read_generated_output(output_dev, cfg):
    start = time.perf_counter()
    _log("  generated: reading and untilizing output")
    tiled = ttnn.to_torch(output_dev).flatten().to(torch.bfloat16)
    expected_numel = cfg.batch * cfg.nheads * cfg.seqlen * cfg.headdim
    if tiled.numel() != expected_numel:
        raise ValueError(
            f"generated output has {tiled.numel()} elements, expected {expected_numel}"
        )
    out = _untilize_nfaces(tiled, rows=cfg.batch * cfg.nheads * cfg.seqlen, cols=cfg.headdim)
    out = out.reshape(cfg.batch, cfg.nheads, cfg.seqlen, cfg.headdim).to(torch.float32)
    _log(f"  generated: output read done in {time.perf_counter() - start:.3f}s")
    return out


def _run_generated_chunk_scan(device, host_inputs, cfg, generated_order):
    start = time.perf_counter()
    _log("  generated: tilizing and moving raw pages to device")
    device_tensors = _prepare_generated_raw_device_tensors(
        device,
        host_inputs,
        cfg,
        with_dst="dst" in generated_order or "out" in generated_order,
    )
    _log(f"  generated: device upload done in {time.perf_counter() - start:.3f}s")

    start = time.perf_counter()
    _log("  generated: launching host_ttnn.run")
    output_dev = _launch_generated_chunk_scan(device_tensors, generated_order)
    ttnn.synchronize_device(device)
    _log(f"  generated: launch/sync done in {time.perf_counter() - start:.3f}s")

    return _read_generated_output(output_dev, cfg)


def _benchmark(device, run_once, warmup, iters, use_trace):
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    if iters <= 0:
        raise ValueError(f"iters must be > 0, got {iters}")

    last_result = None
    for _ in range(warmup):
        last_result = run_once()
    ttnn.synchronize_device(device)
    last_result = None

    profiler.clear()
    if use_trace:
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        for _ in range(iters):
            last_result = run_once()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)

        profiler.start("run")
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        profiler.end("run")
        ttnn.release_trace(device, tid)
        last_result = None
        return profiler.get("run") / iters

    profiler.start("run")
    for _ in range(iters):
        last_result = run_once()
    ttnn.synchronize_device(device)
    profiler.end("run")
    last_result = None
    return profiler.get("run") / iters


def _benchmark_official_ttnn_chunk_scan(device, host_inputs, cfg, warmup, iters, use_trace):
    start = time.perf_counter()
    _log("  official benchmark: moving prepared inputs to device as TTNN TILE tensors")
    device_tensors = _prepare_device_tensors(device, host_inputs, cfg)
    _log(f"  official benchmark: device upload done in {time.perf_counter() - start:.3f}s")

    def run_once():
        return _launch_official_chunk_scan(device_tensors, cfg)

    return _benchmark(
        device=device,
        run_once=run_once,
        warmup=warmup,
        iters=iters,
        use_trace=use_trace,
    )


def _benchmark_generated_chunk_scan(device, host_inputs, cfg, generated_order, warmup, iters, use_trace):
    device_tensors = _prepare_generated_raw_device_tensors(
        device,
        host_inputs,
        cfg,
        with_dst="dst" in generated_order or "out" in generated_order,
    )

    def run_once():
        return _launch_generated_chunk_scan(device_tensors, generated_order)

    return _benchmark(
        device=device,
        run_once=run_once,
        warmup=warmup,
        iters=iters,
        use_trace=use_trace,
    )


def get_causal_flops(batch, seq_len, chunk_size, heads, dim, dstate):
    return (
        2 * batch * seq_len * chunk_size * heads * dim * 0.5
        + 2 * batch * seq_len * heads * dim * dstate
    )


def _report_generated_performance(cfg, grid_size, avg_seconds, warmup, iters, use_trace):
    avg_ms = avg_seconds * 1000.0
    tflops = (
        get_causal_flops(
            batch=cfg.batch,
            seq_len=cfg.seqlen,
            chunk_size=cfg.chunk_size,
            heads=cfg.nheads,
            dim=cfg.headdim,
            dstate=cfg.dstate,
        )
        / avg_seconds
        / 1e12
    )

    _log("Generated chunk-scan performance:")
    _log(f"  Grid: {grid_size}")
    _log(f"  Warmup iterations: {warmup}")
    _log(f"  Measurement iterations: {iters}")
    _log(f"  Trace capture: {'enabled' if use_trace else 'disabled'}")
    _log(f"  Latency: {avg_ms:.6f} ms")
    _log(f"  Coarse causal TFLOPS: {tflops:.6f}")


def _report_speedup(generated_seconds, official_seconds):
    if generated_seconds is None or official_seconds is None:
        return
    if generated_seconds <= 0:
        _log("Speedup generated over official TTNN: unavailable because generated latency is non-positive")
        return
    _log(f"Speedup generated over official TTNN: {official_seconds / generated_seconds:.3f}x")


def _report_official_ttnn_performance(cfg, avg_seconds, warmup, iters, use_trace):
    avg_ms = avg_seconds * 1000.0
    tflops = (
        get_causal_flops(
            batch=cfg.batch,
            seq_len=cfg.seqlen,
            chunk_size=cfg.chunk_size,
            heads=cfg.nheads,
            dim=cfg.headdim,
            dstate=cfg.dstate,
        )
        / avg_seconds
        / 1e12
    )

    _log("Official TTNN chunk-scan performance:")
    _log(f"  Warmup iterations: {warmup}")
    _log(f"  Measurement iterations: {iters}")
    _log(f"  Trace capture: {'enabled' if use_trace else 'disabled'}")
    _log(f"  Latency: {avg_ms:.6f} ms")
    _log(f"  Coarse causal TFLOPS: {tflops:.6f}")


def _pcc(expected, actual):
    expected = expected.flatten().to(torch.float64)
    actual = actual.flatten().to(torch.float64)
    if expected.numel() != actual.numel():
        raise ValueError(f"PCC shape mismatch: {tuple(expected.shape)} vs {tuple(actual.shape)}")
    expected_centered = expected - expected.mean()
    actual_centered = actual - actual.mean()
    denom = torch.linalg.vector_norm(expected_centered) * torch.linalg.vector_norm(actual_centered)
    if denom == 0:
        return float("nan")
    return float(torch.dot(expected_centered, actual_centered) / denom)


def _comparison_stats(expected, actual):
    expected = expected.to(torch.float32)
    actual = actual.to(torch.float32)
    diff = (expected - actual).abs()
    return {
        "pcc": _pcc(expected, actual),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "num_different": int((diff != 0).sum()),
    }


def _get_grid_size(device, grid_x, grid_y):
    hw_grid = device.compute_with_storage_grid_size()
    grid = (
        grid_x if grid_x is not None else int(hw_grid.x),
        grid_y if grid_y is not None else int(hw_grid.y),
    )
    if grid[0] > int(hw_grid.x) or grid[1] > int(hw_grid.y):
        raise ValueError(f"requested grid {grid} exceeds device grid ({int(hw_grid.x)}, {int(hw_grid.y)})")
    return grid


def run_chunk_scan_comparison(args):
    cfg = ChunkScanConfig(
        batch=args.batch,
        seqlen=args.seq_len,
        nheads=args.nhead,
        headdim=args.head_dim,
        ngroups=args.ngroups,
        dstate=args.dstate,
        chunk_size=args.chunk_size,
        block_size=args.block_size,
    )
    _validate_config(cfg)
    generated_order = _resolve_generated_order(args.generated_order)

    _log(
        f"Shape: B={cfg.batch}, H={cfg.nheads}, G={cfg.ngroups}, S={cfg.seqlen}, "
        f"D={cfg.headdim}, DState={cfg.dstate}, chunks={cfg.nchunks}, chunk_size={cfg.chunk_size}, "
        f"block_size={cfg.block_size}"
    )
    _log(f"Generated host_ttnn order: {','.join(generated_order)}")

    device_params = {
        "l1_small_size": args.l1_small_size,
        "trace_region_size": args.trace_region_size,
    }

    host_inputs = _build_inputs(
        cfg,
        low=args.low,
        high=args.high,
        tril_cb=args.tril_cb,
        raw_dt=args.raw_dt,
    )
    references = []
    official_perf_seconds = None
    generated_perf_seconds = None

    if args.with_torch:
        start = time.perf_counter()
        _log("Running torch coarse causal reference...")
        torch_ref = _torch_chunk_scan_reference(host_inputs, cfg, output_bf16=args.bf16_reference)
        _log(f"  torch: reference done in {time.perf_counter() - start:.3f}s")
        references.append(("torch_reference", torch_ref))

    if not args.skip_official:
        official_device = ttnn.CreateDevice(device_id=args.device_id, **device_params)
        try:
            _log("Running official coarse causal TTNN reference...")
            official = _run_official_chunk_scan(official_device, host_inputs, cfg)
            references.append(("official_ttnn", official))
            if not args.skip_perf and args.perf_target in ("official", "both"):
                _log("Benchmarking official TTNN chunk scan implementation...")
                official_perf_seconds = _benchmark_official_ttnn_chunk_scan(
                    device=official_device,
                    host_inputs=host_inputs,
                    cfg=cfg,
                    warmup=args.warmup,
                    iters=args.iters,
                    use_trace=not args.no_trace,
                )
                _report_official_ttnn_performance(
                    cfg=cfg,
                    avg_seconds=official_perf_seconds,
                    warmup=args.warmup,
                    iters=args.iters,
                    use_trace=not args.no_trace,
                )
        finally:
            ttnn.close_device(official_device)

    generated_device = ttnn.CreateDevice(device_id=args.device_id, **device_params)
    try:
        _log("Running generated chunk scan from host_ttnn.py...")
        generated = _run_generated_chunk_scan(generated_device, host_inputs, cfg, generated_order)

        pccs = []
        for label, reference in references:
            stats = _comparison_stats(reference, generated)
            pccs.append(stats["pcc"])
            _log(f"PCC({label}, generated): {stats['pcc']:.9f}")
            _log(f"MaxAbsDiff({label}, generated): {stats['max_abs_diff']:.9f}")
            _log(f"MeanAbsDiff({label}, generated): {stats['mean_abs_diff']:.9f}")
            _log(f"NumDiff({label}, generated): {stats['num_different']}")

        if len(references) >= 2:
            stats = _comparison_stats(references[0][1], references[1][1])
            pccs.append(stats["pcc"])
            _log(f"PCC({references[0][0]}, {references[1][0]}): {stats['pcc']:.9f}")
            _log(f"MaxAbsDiff({references[0][0]}, {references[1][0]}): {stats['max_abs_diff']:.9f}")

        if pccs:
            min_pcc = min(pccs)
            passed = min_pcc >= args.pcc
            _log(f"Result: {'PASS' if passed else 'FAIL'} (min PCC >= {args.pcc})")
        else:
            min_pcc = None
            passed = True
            _log("Result: correctness skipped because no reference path was selected")

        if args.print_values:
            count = min(args.print_values, generated.numel())
            for label, reference in references:
                _log(f"{label}[:{count}] = {reference.flatten()[:count].tolist()}")
            _log(f"generated[:{count}] = {generated.flatten()[:count].tolist()}")

        if not args.skip_perf and args.perf_target in ("mlir", "both"):
            generated_grid = _get_grid_size(generated_device, args.grid_x, args.grid_y)
            _log("Benchmarking generated chunk scan kernel...")
            generated_perf_seconds = _benchmark_generated_chunk_scan(
                device=generated_device,
                host_inputs=host_inputs,
                cfg=cfg,
                generated_order=generated_order,
                warmup=args.warmup,
                iters=args.iters,
                use_trace=not args.no_trace,
            )
            _report_generated_performance(
                cfg=cfg,
                grid_size=generated_grid,
                avg_seconds=generated_perf_seconds,
                warmup=args.warmup,
                iters=args.iters,
                use_trace=not args.no_trace,
            )
            _report_speedup(generated_perf_seconds, official_perf_seconds)

        if args.raise_on_fail and not passed:
            raise AssertionError(f"Minimum PCC {min_pcc} is below threshold {args.pcc}")
    finally:
        ttnn.close_device(generated_device)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare generated chunk-scan host_ttnn.py against the coarse-causal TTNN reference. "
            "Prepared x/output layout is [B, H, S, D]; C layout is [B, G, S, DState]."
        )
    )
    parser.add_argument(
        "shape",
        nargs="?",
        help="Optional shape shorthand like L1024_N32_H128_G4_D128_C128.",
    )
    parser.add_argument(
        "--shape",
        dest="shape_spec",
        type=str,
        default=None,
        help="Shape shorthand like L1024_N32_H128_G4_D128_C128. Explicit dimension flags override it.",
    )
    parser.add_argument("--batch", type=int, default=GENERATED_CONFIG.batch)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--head_dim", type=int, default=None)
    parser.add_argument("--ngroups", type=int, default=None)
    parser.add_argument("--dstate", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=GENERATED_CONFIG.block_size)
    parser.add_argument("--grid_x", type=int, default=8)
    parser.add_argument("--grid_y", type=int, default=8)
    parser.add_argument("--pcc", type=float, default=0.98)
    parser.add_argument("--low", type=float, default=-0.1)
    parser.add_argument("--high", type=float, default=0.1)
    parser.add_argument("--tril_cb", action="store_true")
    parser.add_argument("--raw_dt", action="store_true")
    parser.add_argument("--bf16_reference", action="store_true")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--l1_small_size", type=int, default=24576)
    parser.add_argument("--trace_region_size", type=int, default=7520256 * 20)
    parser.add_argument("--with_torch", action="store_true")
    parser.add_argument("--skip_official", action="store_true")
    parser.add_argument("--print_values", type=int, default=0)
    parser.add_argument("--raise_on_fail", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--no_trace", action="store_true")
    parser.add_argument("--skip_perf", action="store_true")
    parser.add_argument("--perf_target", choices=("official", "mlir", "both"), default="mlir")
    parser.add_argument(
        "--generated_order",
        type=str,
        default="auto",
        help=(
            "Comma-separated host_ttnn.run argument order. "
            "Known names: cb,x,dt,dA,C,prev,D,D_tt,D_host,dst,out. "
            "auto uses cb,x,dt,dA,C,D_host,prev,dst for 8-arg generated host_ttnn.py."
        ),
    )
    return _apply_shape_spec_args(parser.parse_args(), parser)


if __name__ == "__main__":
    run_chunk_scan_comparison(parse_args())
