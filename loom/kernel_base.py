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
    python kernels/matmul.py --config kernels/config.json [--njobs N] [--debug]

    # Or pass paths directly on the command line
    python kernels/matmul.py \\
        --output-path  DIR \\
        --df-mlir      PATH \\
        --hw-compute-dir PATH \\
        [--njobs N] [--debug]
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
    """

    # Override in subclass for a nicer description in --help.
    kernel_name: ClassVar[str] = ""

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
        return _helion_generate_mlir(bound)

    # ------------------------------------------------------------------ #
    # CLI                                                                  #
    # ------------------------------------------------------------------ #

    @classmethod
    def _build_parser(cls) -> argparse.ArgumentParser:
        name = cls.kernel_name or cls.__name__
        parser = argparse.ArgumentParser(
            description=(
                f"Loom pipeline for {name}: "
                "Helion frontend → Explore → ETG resolve → SMT solve → Materialize."
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
            "--df-mlir",
            metavar="MLIR",
            help="Path to DF hardware description MLIR.",
        )
        parser.add_argument(
            "--hw-compute-dir",
            metavar="DIR",
            help="Path to directory containing hardware compute IR (.mlir) files.",
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
            help="Enable detailed SMT analysis and write intermediate IRs/logs.",
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
        df_mlir = args.df_mlir or config_data.get("df_mlir")
        hw_compute_dir = args.hw_compute_dir or config_data.get("hw_compute_dir")
        block_sizes = config_data.get("block_sizes")

        # Required parameter check
        missing = []
        if not output_path:
            missing.append("--output-path (or 'output_path' in config)")
        if not df_mlir:
            missing.append("--df-mlir (or 'df_mlir' in config)")
        if not hw_compute_dir:
            missing.append("--hw-compute-dir (or 'hw_compute_dir' in config)")

        if missing:
            parser.error(f"The following parameters are required: {', '.join(missing)}")

        if block_sizes is not None and not isinstance(block_sizes, dict):
            parser.error("Config 'block_sizes' must be a JSON object mapping symbol names to integers.")

        setup_logging(args.debug)
        run_pipeline(
            generate_mlir_fn=cls.generate_mlir,
            output_path=output_path,
            df_mlir=df_mlir,
            hw_compute_dir=hw_compute_dir,
            njobs=args.njobs,
            debug=args.debug,
            block_sizes=block_sizes,
        )
