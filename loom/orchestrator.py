"""End-to-end Loom pipeline orchestrator.

Runs the full Loom compilation pipeline:
  0. Helion frontend: Python kernel → stage-00 MLIR
  1. Exploration pipeline (stages 0→5): C++ passes via pybind11
  2. SMT solver: finds optimal block sizes per variant
  3. Materialization pipeline (stages 5→7): C++ passes via pybind11

Output layout under --output-path:
  <output-path>/IRs/p00_from_helion_frontend.mlir
  <output-path>/IRs/p01_explored.mlir
  <output-path>/IRs/p02_bufferized.mlir
  <output-path>/constraints/p01_exploration_etg.json
  <output-path>/constraints/smt_solver.log   (when --debug)

NOTE: The SMT solver requires a complete ETG with resolved hardware timing
(produced by the HW analytical model). Until the model is integrated, pass
the pre-computed ETG via --etg-json.

Usage:
    python loom/orchestrator.py \
        --output-path test/mm_2Dmesh \
        --df-mlir     path/to/2D_mesh.mlir \
        --etg-json    path/to/staged_etg_dump.json \
        [--njobs N] \
        [--debug]

Python environment: /opt/miniconda3/envs/loom-dev/bin/python
Required packages: z3-solver, pybind11 (build-time)
"""

import argparse
import json
import sys
from pathlib import Path

from loom_utils.timer import PipelineTimer, print_timing_summary


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Loom end-to-end pipeline: "
            "Helion frontend → Explore → SMT solve → Materialize → OSB → final MLIR."
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
        "--etg-json",
        metavar="JSON",
        help=(
            "Path to the pre-computed ETG JSON (with HW-model resolved_time). "
            "If not provided, uses the exploration-generated ETG "
            "(requires integrated HW analytical model)."
        ),
    )
    parser.add_argument(
        "--njobs",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel SMT solver worker processes (default: 1).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable detailed SMT analysis (active constraints, MUS) and write solver log.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Import the pybind11-based pipeline module and SMT solver.
    # Deferred here so that --help works without a built _loom_pipeline.so.
    from loom_pipeline import run_exploration, run_materialization, smt_run  # noqa: PLC0415

    if smt_run is None:
        print("ERROR: SMT solver not available (loom-dataflow not installed in editable mode).")
        sys.exit(1)

    output_path = Path(args.output_path)
    ir_dir = output_path / "IRs"
    constraints_dir = output_path / "constraints"
    ir_dir.mkdir(parents=True, exist_ok=True)
    constraints_dir.mkdir(parents=True, exist_ok=True)

    p00 = ir_dir / "p00_from_helion_frontend.mlir"
    p01 = ir_dir / "p01_explored.mlir"
    p02 = ir_dir / "p02_bufferized.mlir"
    exploration_etg = constraints_dir / "p01_exploration_etg.json"
    solver_log = constraints_dir / "smt_solver.log" if args.debug else None

    # ETG for the SMT solver: user-supplied (with HW timing) or exploration output
    solver_etg = Path(args.etg_json) if args.etg_json else exploration_etg

    try:
        # ---- Step 0: Helion frontend ----
        print("=" * 72)
        print("STEP 0: HELION FRONTEND (generating stage-00 MLIR)")
        print("=" * 72)
        print()

        with PipelineTimer("Step 0: Helion Frontend"):
            from kernels.matmul import generate_mlir as frontend_generate_mlir  # noqa: PLC0415

            mlir_text = frontend_generate_mlir()
            if args.debug:
                p00.write_text(mlir_text)
                print(f"  Frontend MLIR saved to: {p00}")
        print("\nHelion frontend complete.")

        # ---- Step 1: Exploration pipeline (stages 0→5) ----
        print()
        print("=" * 72)
        print("STEP 1: EXPLORATION PIPELINE (stages 0→5)")
        print("=" * 72)
        print(f"  DF MLIR    : {args.df_mlir}")
        if args.debug:
            print(f"  Output     : {p01}")
        print(f"  ETG output : {exploration_etg}")
        print()

        with PipelineTimer("Step 1: Exploration Pipeline"):
            explored_mlir, etg_json_text = run_exploration(
                input_mlir=mlir_text,
                df_mlir=args.df_mlir,
                produce_etg=True,
            )
            # ETG must be written to disk for the SMT solver.
            exploration_etg.write_text(etg_json_text)
            if args.debug:
                p01.write_text(explored_mlir)

        # Free frontend MLIR — no longer needed.
        del mlir_text

        print("Exploration pipeline complete.")

        # ---- Step 2: SMT solver ----
        print()
        print("=" * 72)
        print("STEP 2: SMT SOLVER")
        print("=" * 72)
        print(f"  ETG input  : {solver_etg}")
        print()

        smt_args = argparse.Namespace(
            input=solver_etg,
            njobs=args.njobs,
            output=solver_log,
            debug=args.debug,
        )

        with PipelineTimer("Step 2: SMT Solver"):
            block_sizes = smt_run(smt_args)

        feasible_count = sum(1 for v in block_sizes.values() if v is not None)
        if feasible_count == 0:
            print("\nERROR: All variants UNSAT. No feasible block sizes. Aborting.")
            sys.exit(1)

        print(f"\nSolver found feasible block sizes for {feasible_count} variant(s).")

        # ---- Step 3: Materialization pipeline (stages 5→7) ----
        print()
        print("=" * 72)
        print("STEP 3: MATERIALIZATION PIPELINE (Materialize → OSB)")
        print("=" * 72)
        print(f"  Output : {p02}")
        print()

        with PipelineTimer("Step 3: Materialization"):
            final_mlir = run_materialization(
                input_mlir=explored_mlir,
                block_sizes_json=json.dumps(block_sizes),
            )

        # Free explored MLIR — no longer needed.
        del explored_mlir

        # Always write the final output.
        p02.write_text(final_mlir)

        print(f"Pipeline complete. Final MLIR written to: {p02}")
        print_timing_summary()

    except Exception:
        raise


if __name__ == "__main__":
    main()
