"""End-to-end Loom pipeline orchestrator.

Runs the full Loom compilation pipeline:
  0. Helion frontend: Python kernel → stage-00 MLIR
  1. Exploration pipeline (stages 0→5): C++ passes via pybind11
  2. ETG resolution: fill scenarios via MLAR Rust evaluator
  3. SMT solver: finds optimal block sizes per variant
  4. Materialization pipeline (stages 5→7): C++ passes via pybind11

Output layout under --output-path:
  <output-path>/IRs/p00_from_helion_frontend.mlir   (when --debug)
  <output-path>/IRs/p01_explored.mlir               (when --debug)
  <output-path>/IRs/p03_bufferized.mlir
  <output-path>/constraints/p01_exploration_etg.json
  <output-path>/constraints/p02_resolved_etg.json
  <output-path>/constraints/smt_solver.log           (when --debug)

Usage:
    python loom/orchestrator.py \
        --output-path    test/mm_2Dmesh \
        --df-mlir        path/to/2D_mesh.mlir \
        --hw-compute-dir path/to/compute/ \
        [--njobs N] \
        [--debug]

Python environment: /opt/miniconda3/envs/loom-dev/bin/python
Required packages: z3-solver, pybind11 (build-time)
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from loom_utils.timer import pipeline_timer, print_timing_summary

# ---------------------------------------------------------------------------
# Logging configuration
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
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Loom end-to-end pipeline: "
            "Helion frontend → Explore → ETG resolve → SMT solve → Materialize → OSB → final MLIR."
        )
    )
    parser.add_argument(
        "--output-path",
        required=True,
        metavar="DIR",
        help="Root output directory. MLIRs go to <DIR>/IRs/, logs/JSON to <DIR>/constraints/.",
    )
    parser.add_argument(
        "--df-mlir",
        required=True,
        metavar="MLIR",
        help="Path to DF hardware description MLIR (e.g. 2D_mesh.mlir).",
    )
    parser.add_argument(
        "--hw-compute-dir",
        required=True,
        metavar="DIR",
        help="Path to directory containing hardware compute IR (.mlir) files.",
    )
    parser.add_argument(
        "--njobs",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel workers for ETG resolution and SMT solving (default: 1).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable detailed SMT analysis (active constraints, MUS) and write solver log.",
    )
    return parser


# ---------------------------------------------------------------------------
# Pipeline Steps
# ---------------------------------------------------------------------------

def run_step_0_frontend(ir_dir: Path, debug: bool) -> str:
    """Step 0: Helion frontend (Python kernel → stage-00 MLIR)."""
    logging.info("=" * 72)
    logging.info("STEP 0: HELION FRONTEND (generating stage-00 MLIR)")
    logging.info("=" * 72)

    with pipeline_timer("Step 0: Helion Frontend"):
        from kernels.matmul import generate_mlir as frontend_generate_mlir  # noqa: PLC0415

        mlir_text = frontend_generate_mlir()
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

    from loom_utils.mlar_evaluator import resolve_etg_variants, smart_json_dumps  # noqa: PLC0415

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
) -> dict[str, Any]:
    """Step 3: SMT solver (finds optimal block sizes)."""
    logging.info("")
    logging.info("=" * 72)
    logging.info("STEP 3: SMT SOLVER")
    logging.info("=" * 72)
    logging.info(f"  ETG input  : {resolved_etg_path}")

    from smt import smt_run  # noqa: PLC0415

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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    setup_logging(args.debug)

    output_path = Path(args.output_path)
    ir_dir = output_path / "IRs"
    constraints_dir = output_path / "constraints"
    ir_dir.mkdir(parents=True, exist_ok=True)
    constraints_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: Helion frontend
    mlir_text = run_step_0_frontend(ir_dir, args.debug)

    # Step 1: Exploration pipeline
    explored_mlir, etg_json_text = run_exploration_step(
        mlir_text, args.df_mlir, args.hw_compute_dir, ir_dir, constraints_dir, args.debug
    )
    del mlir_text  # Free memory

    # Step 2: ETG resolution
    run_step_2_etg_resolution(etg_json_text, args.njobs, constraints_dir)
    del etg_json_text  # Free memory

    # Step 3: SMT Solver
    resolved_etg_path = constraints_dir / "p02_resolved_etg.json"
    block_sizes = run_step_3_smt_solve(resolved_etg_path, args.njobs, args.debug, constraints_dir)

    # Step 4: Materialization
    run_step_4_materialization(explored_mlir, block_sizes, ir_dir)
    del explored_mlir  # Free memory

    print_timing_summary()


# Backward compatibility for Step 1 function name
run_exploration_step = run_step_1_exploration


if __name__ == "__main__":
    main()
