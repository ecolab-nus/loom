"""LoomKernel — base class for Loom kernel scripts.

Subclassing ``LoomKernel`` gives a kernel module a full CLI plus the
complete Loom compilation pipeline at zero extra cost.

Minimal usage
-------------
::

    import torch
    import helion
    import helion.language as hl
    from loom import LoomKernel

    class Matmul(LoomKernel):
        # --- kernel dimensions (can be overridden per-subclass) ---
        M, K, N = 4096, 512, 4096

        # Assign the helion-decorated function as a class attribute.
        # We cannot stack @staticmethod and @helion.kernel because helion
        # returns a custom object, not a plain function.
        @staticmethod
        def _matmul_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            ...  # helion body

        kernel = helion.kernel(static_shapes=False)(_matmul_fn.__func__)

        @classmethod
        def bind_args(cls):
            x = torch.randn([cls.M, cls.K], device="cpu", dtype=torch.float16)
            y = torch.randn([cls.K, cls.N], device="cpu", dtype=torch.float16)
            return (x, y)

    if __name__ == "__main__":
        Matmul.run()

CLI produced (inherited from LoomKernel)
----------------------------------------
::

    # Recommended: load paths from a config file
    python kernels/matmul.py --config kernels/config.json [--njobs N] [--debug] \
        [--topk-candidates K] [--topk-block-size K]

    # Or pass paths directly on the command line
    python kernels/matmul.py \
        --output-path  DIR \
        --hw-spec      PATH \
        [--njobs N] [--debug] [--topk-candidates K] [--topk-block-size K]

Config file notes
-----------------
::

    {
      "output_path": "...",
      "hw_spec": "...",
      "assigned_block_size": {
        "variant_name": {"tile_m": 512, "tile_n": 512, "tile_k": 64}
      }
    }

    # Or materialize multiple block-size combos for the same variant:
    {
      "assigned_block_size": {
        "variant_name": [
          {"tile_m": 512, "tile_n": 512, "tile_k": 64},
          {"tile_m": 256, "tile_n": 256, "tile_k": 128}
        ]
      }
    }

When ``assigned_block_size`` is a non-empty JSON object, Loom skips the
solver and uses these values directly for materialization. In debug mode,
ETG generation/resolution still runs so Loom can write manual latency
breakdowns to ``constraints/solver.log``.
"""
from __future__ import annotations

import argparse
import sys
from typing import ClassVar

from loom.pipeline import run_pipeline, setup_logging


class LoomKernel:
    """Abstract base for Loom kernel scripts.

    Subclasses **must** define:

    ``kernel``
        A helion-decorated callable (class attribute).
        Assign it as ``kernel = helion.kernel(...)(fn)`` rather than
        stacking ``@staticmethod @helion.kernel`` because the helion
        decorator returns a custom object, not a plain function.

    ``bind_args() -> tuple``
        A classmethod returning the concrete tensors (or other args)
        to pass to ``kernel.bind()``.  These define the shapes,
        dtypes, and devices for MLIR generation.

    Optional overrides:

    ``kernel_name: str``
        Human-readable name shown in CLI help text.  Defaults to the
        class name.

    ``assume_divisible_tiles: bool``
        Passed to ``helion_mlir.generate_mlir``. When true, lowering assumes
        tile bounds are divisible and may omit dynamic tail handling.
    """

    # Override in subclass for a nicer description in --help.
    kernel_name: ClassVar[str] = ""
    # Override per kernel to control Helion MLIR lowering behavior.
    assume_divisible_tiles: ClassVar[bool] = False

    # ------------------------------------------------------------------ #
    # Subclass interface                                                   #
    # ------------------------------------------------------------------ #

    kernel = None   # must be set by subclass: helion-decorated callable

    @classmethod
    def bind_args(cls) -> tuple:
        """Return a tuple of concrete tensors to bind the kernel with."""
        raise NotImplementedError(
            f"{cls.__name__}.bind_args() not implemented. "
            "Override this classmethod to provide concrete tensor shapes."
        )

    # ------------------------------------------------------------------ #
    # MLIR generation                                                      #
    # ------------------------------------------------------------------ #

    @classmethod
    def generate_mlir(cls) -> str:
        """Bind the kernel to concrete tensors and return stage-00 MLIR text.

        This is the callable injected into the pipeline as ``generate_mlir_fn``.
        """
        from helion_mlir import generate_mlir as _helion_generate_mlir  # noqa: PLC0415
        from helion_mlir import print_debug_info  # noqa: PLC0415

        if cls.kernel is None:
            raise RuntimeError(
                f"{cls.__name__}.kernel is None. "
                "Set 'kernel = helion.kernel(...)(fn)' as a class attribute."
            )

        args = cls.bind_args()
        bound = cls.kernel.bind(args)
        print_debug_info(bound)
        return _helion_generate_mlir(
            bound,
            assume_divisible_tiles=cls.assume_divisible_tiles,
        )

    # ------------------------------------------------------------------ #
    # CLI                                                                  #
    # ------------------------------------------------------------------ #

    @classmethod
    def _build_parser(cls) -> argparse.ArgumentParser:
        name = cls.kernel_name or cls.__name__
        parser = argparse.ArgumentParser(
            description=(
                f"Loom pipeline for {name}: "
                "Helion frontend → Explore → ETG resolve → CP-SAT solve → Materialize."
            )
        )
        parser.add_argument(
            "--config",
            metavar="JSON",
            help="Path to configuration JSON file providing defaults for output and hardware paths.",
        )
        parser.add_argument(
            "--output-path",
            metavar="DIR",
            help="Root output directory.",
        )
        parser.add_argument(
            "--hw-spec",
            metavar="MLIR",
            help="Path to hardware specification MLIR (topology + components).",
        )
        parser.add_argument(
            "--njobs",
            type=int,
            default=1,
            metavar="N",
            help="Number of parallel workers (default: 1).",
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            default=False,
            help="Enable detailed analysis and write intermediate IRs/logs.",
        )
        parser.add_argument(
            "--topk-candidates",
            dest="topk_candidates",
            type=_positive_int,
            default=None,
            metavar="K",
            help="Keep only the top K candidates by optimal time in the final output.",
        )
        parser.add_argument(
            "--topk-block-size",
            type=_positive_odd_int,
            default=1,
            metavar="K",
            help="Materialize K local block-size samples per symbol around each selected candidate.",
        )
        return parser

    @classmethod
    def run(cls) -> None:
        """Parse CLI arguments and execute the full Loom pipeline.

        Call this from ``if __name__ == '__main__'`` in the kernel script.
        """
        import json
        from pathlib import Path

        parser = cls._build_parser()
        args = parser.parse_args()

        # Load config if provided
        config_data = {}
        if args.config:
            config_path = Path(args.config)
            if not config_path.exists():
                parser.error(f"Config file not found: {args.config}")
            with config_path.open() as f:
                try:
                    config_data = json.load(f)
                except json.JSONDecodeError as e:
                    parser.error(f"Failed to parse config JSON: {e}")

        # Resolve parameters: CLI takes precedence over config
        output_path = args.output_path or config_data.get("output_path")
        hw_spec = args.hw_spec or config_data.get("hw_spec") or config_data.get("df_mlir")
        block_sizes = config_data.get("block_sizes")
        assigned_block_size = config_data.get("assigned_block_size")

        # Required parameter check
        missing = []
        if not output_path:
            missing.append("--output-path (or 'output_path' in config)")
        if not hw_spec:
            missing.append("--hw-spec (or 'hw_spec' in config)")

        if missing:
            parser.error(f"The following parameters are required: {', '.join(missing)}")

        if assigned_block_size is not None and not isinstance(assigned_block_size, dict):
            parser.error("Config 'assigned_block_size' must be a JSON object.")
        has_assigned_block_size = bool(assigned_block_size)

        symbol_domains = None
        if block_sizes is not None:
            if not isinstance(block_sizes, dict):
                parser.error("Config 'block_sizes' must be a JSON object.")
            for sym, bounds in block_sizes.items():
                if not isinstance(bounds, dict) or "lb" not in bounds or "ub" not in bounds:
                    parser.error(
                        f"block_sizes['{sym}'] must be a dict with 'lb' and 'ub' keys."
                    )
                if bounds["lb"] > bounds["ub"]:
                    parser.error(
                        f"block_sizes['{sym}']: lb ({bounds['lb']}) must be <= ub ({bounds['ub']})."
                    )
            from loom.loom_utils.modeling import parse_user_block_sizes  # noqa: PLC0415
            symbol_domains = parse_user_block_sizes(block_sizes)

        setup_logging(args.debug)
        run_pipeline(
            generate_mlir_fn=cls.generate_mlir,
            output_path=output_path,
            hw_spec=hw_spec,
            njobs=args.njobs,
            debug=args.debug,
            symbol_domains=symbol_domains,
            assigned_block_size=assigned_block_size if has_assigned_block_size else None,
            topk_candidates=args.topk_candidates,
            topk_block_size=args.topk_block_size,
        )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_odd_int(value: str) -> int:
    parsed = _positive_int(value)
    if parsed % 2 == 0:
        raise argparse.ArgumentTypeError("value must be an odd integer")
    return parsed
