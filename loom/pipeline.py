"""Loom end-to-end compilation pipeline — library interface.

This module is the sole owner of all pipeline logic. It exposes:

    run_pipeline(generate_mlir_fn, *, output_path, df_mlir,
                 hw_compute_dir, njobs, debug)

where ``generate_mlir_fn`` is any callable that returns stage-00 MLIR
text.  The function is injected by the caller (usually a ``LoomKernel``
subclass), so the pipeline has **no dependency on any kernel module**.

Pipeline stages
---------------
  0. Helion frontend   – call ``generate_mlir_fn()`` → raw MLIR text
  1. Exploration        – C++ passes via pybind11 (loom_pipeline)
  2. ETG resolution     – MLAR Rust evaluator (loom_utils)
  3. SMT solver         – Z3-based block-size optimizer (loom.smt)
  4. Materialization    – C++ passes via pybind11 (loom_pipeline)

Output layout under <output_path>
----------------------------------
  IRs/p00_from_helion_frontend.mlir   (--debug only)
  IRs/p01_explored.mlir               (--debug only)
  IRs/p03_bufferized.mlir
  constraints/p01_exploration_etg.json
  constraints/p02_resolved_etg.json
  constraints/smt_solver.log          (--debug only)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

from loom.loom_utils.timer import pipeline_timer, print_timing_summary


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(debug: bool = False) -> None:
    """Configure structured logging for the Loom pipeline."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def run_step_0_frontend(
    generate_mlir_fn: Callable[[], str],
    ir_dir: Path,
    debug: bool,
) -> str:
    """Step 0: Call the kernel's generate_mlir callable to produce stage-00 MLIR."""
    logging.info("=" * 72)
    logging.info("STEP 0: HELION FRONTEND (generating stage-00 MLIR)")
    logging.info("=" * 72)

    with pipeline_timer("Step 0: Helion Frontend"):
        mlir_text = generate_mlir_fn()
        if debug:
            p00 = ir_dir / "p00_from_helion_frontend.mlir"
            p00.write_text(mlir_text)
            logging.info(f"  Frontend MLIR saved to: {p00}")

    logging.info("Helion frontend complete.")
    return mlir_text


def run_step_1_exploration(
    mlir_text: str,
    df_mlir: str,
    hw_compute_dir: str,
    ir_dir: Path,
    constraints_dir: Path,
    debug: bool,
) -> tuple[str, str]:
    """Step 1: Exploration pipeline (stages 0→5) via pybind11."""
    logging.info("")
    logging.info("=" * 72)
    logging.info("STEP 1: EXPLORATION PIPELINE (stages 0→5)")
    logging.info("=" * 72)
    logging.info(f"  DF MLIR    : {df_mlir}")

    from loom_pipeline import run_exploration  # noqa: PLC0415

    p01 = ir_dir / "p01_explored.mlir"
    exploration_etg = constraints_dir / "p01_exploration_etg.json"

    if debug:
        logging.info(f"  Output     : {p01}")
    logging.info(f"  ETG output : {exploration_etg}")

    with pipeline_timer("Step 1: Exploration Pipeline"):
        explored_mlir, etg_json_text = run_exploration(
            input_mlir=mlir_text,
            df_mlir=df_mlir,
            hw_compute_dir=hw_compute_dir,
            produce_etg=True,
        )
        exploration_etg.write_text(etg_json_text)
        if debug:
            p01.write_text(explored_mlir)

    logging.info("Exploration pipeline complete.")
    return explored_mlir, etg_json_text


def run_step_2_etg_resolution(
    etg_json_text: str,
    njobs: int,
    constraints_dir: Path,
) -> list[dict[str, Any]]:
    """Step 2: ETG resolution via MLAR Rust evaluator."""
    logging.info("")
    logging.info("=" * 72)
    logging.info("STEP 2: ETG RESOLUTION (MLAR evaluator)")
    logging.info("=" * 72)

    from loom.loom_utils.mlar_evaluator import resolve_etg_variants, smart_json_dumps  # noqa: PLC0415

    resolved_etg = constraints_dir / "p02_resolved_etg.json"
    logging.info(f"  Output : {resolved_etg}")

    with pipeline_timer("Step 2: ETG Resolution"):
        variants = json.loads(etg_json_text)
        resolved_variants = resolve_etg_variants(variants, njobs=njobs)
        resolved_etg.write_text(smart_json_dumps(resolved_variants))

    logging.info(f"ETG resolution complete. {len(resolved_variants)} variant(s) resolved.")
    return resolved_variants


def run_step_3_smt_solve(
    resolved_etg_path: Path,
    njobs: int,
    debug: bool,
    constraints_dir: Path,
    force_enumerate: bool = False,
) -> dict[str, Any]:
    """Step 3: SMT solver (finds optimal block sizes)."""
    logging.info("")
    logging.info("=" * 72)
    logging.info("STEP 3: SMT SOLVER")
    logging.info("=" * 72)
    logging.info(f"  ETG input  : {resolved_etg_path}")

    from loom.smt import smt_run  # noqa: PLC0415

    if smt_run is None:
        logging.error("SMT solver not available (loom-dataflow not installed in editable mode).")
        sys.exit(1)

    solver_log = constraints_dir / "smt_solver.log" if debug else None

    with pipeline_timer("Step 3: SMT Solver"):
        block_sizes = smt_run(
            input_path=resolved_etg_path,
            njobs=njobs,
            output_path=solver_log,
            debug=debug,
            force_enumerate=force_enumerate,
        )

    feasible_count = sum(1 for v in block_sizes.values() if v is not None)
    if feasible_count == 0:
        logging.error("All variants UNSAT. No feasible block sizes. Aborting.")
        sys.exit(1)

    logging.info(f"Solver found feasible block sizes for {feasible_count} variant(s).")
    return block_sizes


def run_step_4_materialization(
    explored_mlir: str,
    block_sizes: dict[str, Any],
    ir_dir: Path,
) -> None:
    """Step 4: Materialization pipeline (stages 5→7) via pybind11."""
    logging.info("")
    logging.info("=" * 72)
    logging.info("STEP 4: MATERIALIZATION PIPELINE (Materialize → OSB)")
    logging.info("=" * 72)

    from loom_pipeline import run_materialization  # noqa: PLC0415

    p03 = ir_dir / "p03_bufferized.mlir"
    logging.info(f"  Output : {p03}")

    with pipeline_timer("Step 4: Materialization"):
        final_mlir = run_materialization(
            input_mlir=explored_mlir,
            block_sizes_json=json.dumps(block_sizes),
        )
        p03.write_text(final_mlir)

    logging.info(f"Pipeline complete. Final MLIR written to: {p03}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_and_broadcast_block_sizes(
    manual_block_sizes: dict[str, int],
    etg_json_text: str,
) -> dict[str, dict[str, int]]:
    """Validate that manual block sizes cover all ETG symbols and broadcast to all variants.

    Parameters
    ----------
    manual_block_sizes:
        A single dictionary of {symbol_name: value}.
    etg_json_text:
        The raw ETG JSON string from Step 1.

    Returns
    -------
    A broadcast dict compatible with Step 4: {variant_name: manual_block_sizes}.
    """
    variants = json.loads(etg_json_text)
    if not variants:
        return {}

    # Extract all unique symbols required by any variant
    required_symbols = set()
    for v in variants:
        symbols = v.get("constraint_scope", {}).get("metadata", {}).get("symbols", {})
        required_symbols.update(symbols.keys())

    # Guard: check coverage
    missing = [s for s in required_symbols if s not in manual_block_sizes]
    if missing:
        raise ValueError(
            f"Manual block_sizes are missing values for the following symbols "
            f"required by the ETG: {', '.join(sorted(missing))}"
        )

    # Broadcast
    return {
        v.get("variant_name", f"variant_{i}"): manual_block_sizes
        for i, v in enumerate(variants)
    }


# ---------------------------------------------------------------------------
# Public API — single entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    generate_mlir_fn: Callable[[], str],
    *,
    output_path: str | Path,
    df_mlir: str | Path,
    hw_compute_dir: str | Path,
    njobs: int = 1,
    debug: bool = False,
    block_sizes: dict[str, int] | None = None,
    force_enumerate: bool = False,
) -> None:
    """Run the full Loom compilation pipeline.

    Parameters
    ----------
    generate_mlir_fn:
        A zero-argument callable that returns the stage-00 MLIR text for
        the kernel.  Typically ``MyKernel.generate_mlir``.
    output_path:
        Root output directory.  Sub-directories ``IRs/`` and
        ``constraints/`` are created automatically.
    df_mlir:
        Path to the DF hardware description MLIR file.
    hw_compute_dir:
        Path to the directory containing hardware compute IR (.mlir) files.
    njobs:
        Number of parallel workers for ETG resolution and SMT solving.
    debug:
        Enable detailed SMT analysis and write intermediate IRs/logs.
    block_sizes:
        Optional manual block size overrides. If provided, Steps 2 and 3
        (ETG resolution and SMT solving) are skipped.
    """
    output_path = Path(output_path)
    ir_dir = output_path / "IRs"
    constraints_dir = output_path / "constraints"
    ir_dir.mkdir(parents=True, exist_ok=True)
    constraints_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: Helion frontend
    mlir_text = run_step_0_frontend(generate_mlir_fn, ir_dir, debug)

    # Step 1: Exploration
    explored_mlir, etg_json_text = run_step_1_exploration(
        mlir_text, str(df_mlir), str(hw_compute_dir), ir_dir, constraints_dir, debug
    )
    del mlir_text

    # Step 2: ETG resolution & Step 3: SMT Solver
    if block_sizes is not None:
        logging.info("")
        logging.info("⏭  block_sizes provided in config — skipping Steps 2 & 3.")
        broadcast = validate_and_broadcast_block_sizes(block_sizes, etg_json_text)
    else:
        # Step 2: ETG resolution
        run_step_2_etg_resolution(etg_json_text, njobs, constraints_dir)

        # Step 3: SMT Solver
        resolved_etg_path = constraints_dir / "p02_resolved_etg.json"
        broadcast = run_step_3_smt_solve(resolved_etg_path, njobs, debug, constraints_dir, force_enumerate)

    del etg_json_text

    # Step 4: Materialization
    run_step_4_materialization(explored_mlir, broadcast, ir_dir)
    del explored_mlir

    print_timing_summary()
