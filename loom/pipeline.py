"""Loom end-to-end compilation pipeline — library interface.

This module is the sole owner of all pipeline logic. It exposes:

    run_pipeline(generate_mlir_fn, *, output_path, hw_spec,
                 njobs, debug)

where ``generate_mlir_fn`` is any callable that returns stage-00 MLIR
text.  The function is injected by the caller (usually a ``LoomKernel``
subclass), so the pipeline has **no dependency on any kernel module**.

Pipeline stages
---------------
  0. Helion frontend   – call ``generate_mlir_fn()`` → raw MLIR text
  1. Exploration        – C++ passes via pybind11 (loom_pipeline)
  2. ETG resolution     – MLAR Rust evaluator (loom_utils, optional)
  3. CP-SAT solver      – CPMpy/OR-Tools block-size optimizer (optional)
  4. Materialization    – C++ passes via pybind11 (loom_pipeline)

When ``assigned_block_size`` is provided (non-empty), solving is bypassed.
ETG generation/resolution is still run in debug mode so manual latency
breakdowns can be written to ``solver.log``.

Output layout under <output_path>
----------------------------------
  IRs/p00_from_helion_frontend.mlir   (--debug only)
  IRs/p01_explored.mlir               (--debug only)
  IRs/p03_bufferized.mlir
  constraints/p01_exploration_etg.json
  constraints/p02_resolved_etg.json
  constraints/solver.log              (--debug only)
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
    hw_spec: str,
    ir_dir: Path,
    constraints_dir: Path,
    debug: bool,
    skip_etg: bool = False,
) -> tuple[str, str]:
    """Step 1: Exploration pipeline (stages 0→5) via pybind11.

    If ``skip_etg`` is True, ETG generation is disabled and no ETG file is written.
    """
    logging.info("")
    logging.info("=" * 72)
    logging.info("STEP 1: EXPLORATION PIPELINE (stages 0→5)")
    logging.info("=" * 72)
    logging.info(f"  Hardware Spec: {hw_spec}")

    from loom_pipeline import run_exploration  # noqa: PLC0415

    p01 = ir_dir / "p01_explored.mlir"
    exploration_etg = constraints_dir / "p01_exploration_etg.json"

    if debug:
        logging.info(f"  Output     : {p01}")
    if skip_etg:
        logging.info("  ETG output : [SKIPPED via assigned_block_size override]")
    else:
        logging.info(f"  ETG output : {exploration_etg}")

    with pipeline_timer("Step 1: Exploration Pipeline"):
        explored_mlir, etg_json_text = run_exploration(
            input_mlir=mlir_text,
            hw_spec_file=hw_spec,
            produce_etg=True,
            skip_etg=skip_etg,
            # spatial_reuse=False
        )
        if not skip_etg:
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

    from loom.loom_utils.mlar import resolve_etg_variants  # noqa: PLC0415
    from loom.loom_utils.io import smart_json_dumps  # noqa: PLC0415

    resolved_etg = constraints_dir / "p02_resolved_etg.json"
    logging.info(f"  Output : {resolved_etg}")

    with pipeline_timer("Step 2: ETG Resolution"):
        variants = json.loads(etg_json_text)
        resolved_variants = resolve_etg_variants(variants, njobs=njobs)
        resolved_etg.write_text(smart_json_dumps(resolved_variants))

    logging.info(f"ETG resolution complete. {len(resolved_variants)} variant(s) resolved.")
    return resolved_variants


def run_step_3_solve(
    resolved_etg_path: Path,
    njobs: int,
    debug: bool,
    constraints_dir: Path,
    symbol_domains: dict[str, list[int]] | None = None,
    topk_candidates: int | None = None,
    topk_block_size: int = 1,
) -> dict[str, Any]:
    """Step 3: CPMpy/CP-SAT solver (finds optimal block sizes)."""
    if topk_candidates is not None and topk_candidates <= 0:
        raise ValueError("topk_candidates must be a positive integer")
    if topk_block_size <= 0:
        raise ValueError("topk_block_size must be a positive integer")

    logging.info("")
    logging.info("=" * 72)
    logging.info("STEP 3: SOLVER (CPMpy/CP-SAT)")
    logging.info("=" * 72)
    logging.info(f"  ETG input  : {resolved_etg_path}")

    from loom.solver import cpmpy_run  # noqa: PLC0415

    solver_log = constraints_dir / "solver.log" if debug else None

    with pipeline_timer("Step 3: Solver"):
        block_sizes = cpmpy_run(
            input_path=resolved_etg_path,
            njobs=njobs,
            output_path=solver_log,
            symbol_domains=symbol_domains,
            topk_candidates=topk_candidates,
            topk_block_size=topk_block_size,
            debug=debug,
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
        materialization_input = {
            "__loom_candidate_order__": list(block_sizes),
            **block_sizes,
        }
        final_mlir = run_materialization(
            input_mlir=explored_mlir,
            block_sizes_json=json.dumps(materialization_input),
        )
        p03.write_text(final_mlir)

    logging.info(f"Pipeline complete. Final MLIR written to: {p03}")


# ---------------------------------------------------------------------------
# Public API — single entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    generate_mlir_fn: Callable[[], str],
    *,
    output_path: str | Path,
    hw_spec: str | Path,
    njobs: int = 1,
    debug: bool = False,
    symbol_domains: dict[str, list[int]] | None = None,
    assigned_block_size: dict[str, Any] | None = None,
    topk_candidates: int | None = None,
    topk_block_size: int = 1,
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
    hw_spec:
        Path to the hardware specification MLIR file.
    njobs:
        Number of parallel workers for ETG resolution and CP-SAT solving.
    debug:
        Enable detailed analysis and write intermediate IRs/logs.
    symbol_domains:
        Optional per-symbol domain overrides mapping symbol name to a list
        of allowed values.  If provided, the solver uses these domains
        instead of the built-in defaults.  Steps 2 and 3 still run in all cases.
    assigned_block_size:
        Optional explicit block-size assignments to use directly for
        materialization. When provided as a non-empty dict, Step 3 is skipped.
        Step 2 also runs in debug mode to support manual latency reporting.
    topk_candidates:
        Optional positive integer limiting materialization to the top K
        candidates by optimal time.
    topk_block_size:
        Optional positive odd integer controlling 32-step neighbor sampling
        around each solver-selected block-size assignment.
    """
    output_path = Path(output_path)
    ir_dir = output_path / "IRs"
    constraints_dir = output_path / "constraints"
    ir_dir.mkdir(parents=True, exist_ok=True)
    constraints_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: Helion frontend
    mlir_text = run_step_0_frontend(generate_mlir_fn, ir_dir, debug)

    has_assigned_block_size = bool(assigned_block_size)
    needs_manual_etg = (
        has_assigned_block_size
        and (debug or "ALL" in assigned_block_size)
    )

    # Step 1: Exploration
    explored_mlir, etg_json_text = run_step_1_exploration(
        mlir_text, str(hw_spec), ir_dir, constraints_dir, debug,
        skip_etg=has_assigned_block_size and not needs_manual_etg,
    )
    del mlir_text

    if has_assigned_block_size:
        if needs_manual_etg:
            resolved_variants = run_step_2_etg_resolution(
                etg_json_text, njobs, constraints_dir
            )
            from loom.solver import (  # noqa: PLC0415
                prepare_manual_block_sizes,
                write_manual_breakdown_log,
            )

            if debug:
                solver_log = constraints_dir / "solver.log"
                block_size = write_manual_breakdown_log(
                    resolved_variants,
                    assigned_block_size,
                    solver_log,
                )
                logging.info(f"Manual latency breakdown written to: {solver_log}")
            else:
                block_size = prepare_manual_block_sizes(
                    resolved_variants,
                    assigned_block_size,
                )
        else:
            logging.info("")
            logging.info("=" * 72)
            logging.info("STEP 2: ETG RESOLUTION (MLAR evaluator)")
            logging.info("=" * 72)
            logging.info("Skipped due to assigned_block_size override.")
            block_size = assigned_block_size

        logging.info("")
        logging.info("=" * 72)
        logging.info("STEP 3: SOLVER (CPMpy/CP-SAT)")
        logging.info("=" * 72)
        logging.info("Skipped due to assigned_block_size override.")
    else:
        # Step 2: ETG resolution
        run_step_2_etg_resolution(etg_json_text, njobs, constraints_dir)

        # Step 3: Solver
        resolved_etg_path = constraints_dir / "p02_resolved_etg.json"
        block_size = run_step_3_solve(
            resolved_etg_path, njobs, debug, constraints_dir,
            symbol_domains=symbol_domains,
            topk_candidates=topk_candidates,
            topk_block_size=topk_block_size,
        )

    del etg_json_text

    # Step 4: Materialization
    run_step_4_materialization(explored_mlir, block_size, ir_dir)
    del explored_mlir

    if debug:
        print_timing_summary()
