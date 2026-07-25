#!/bin/bash
# One-click developer install for the Loom monorepo.
#
# Prepares the core pipeline development environment:
#   1. Git submodule init/update
#   2. Pre-flight system dependency checks
#   3. Stage 1: uv sync (Python dependencies and Helion)
#   4. Stage 2: bundled LLVM/MLIR, loom-dataflow, and loom-mlar
#
# Usage:
#   bash install-dev.sh [OPTIONS]
#
# Options:
#   --skip-mlar         Skip building the loom-mlar Rust evaluator
#   --skip-dataflow     Skip building loom-dataflow (C++ MLIR passes)
#   --skip-helion       Skip installing helion-mlir
#   --rebuild-dataflow  Force rebuilding loom-dataflow
#   --help              Show this help message
#
# Environment:
#   LOOM_LLVM_JOBS      Parallel jobs for the bundled LLVM/MLIR build
#   LOOM_EVAL_SYSTEM    Path to pre-built eval_system binary (skips Rust build)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LLVM_BUILD_DIR="$REPO_ROOT/build/llvm-4efe170d"
export MLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir"
SKIP_MLAR=0
SKIP_DATAFLOW=0
SKIP_HELION=0
REBUILD_DATAFLOW=0

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --skip-mlar)        SKIP_MLAR=1 ;;
        --skip-dataflow)    SKIP_DATAFLOW=1 ;;
        --skip-helion)      SKIP_HELION=1 ;;
        --rebuild-dataflow) REBUILD_DATAFLOW=1 ;;
        --help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $arg (see --help)"
            exit 1
            ;;
    esac
done

export SKIP_DATAFLOW
export SKIP_MLAR

# ---------------------------------------------------------------------------
# Step 0: Git submodules
# ---------------------------------------------------------------------------
echo ""
echo "=== Initializing git submodules ==="
git -C "$REPO_ROOT" submodule update --init --recursive

# ---------------------------------------------------------------------------
# Step 1: Pre-flight checks
# ---------------------------------------------------------------------------
echo ""
echo "=== Pre-flight dependency checks ==="
export REPO_ROOT
source "$REPO_ROOT/scripts/preflight.sh"
if ! run_preflight_checks; then
    echo ""
    echo "Aborting due to pre-flight errors."
    exit 1
fi

# ---------------------------------------------------------------------------
# Stage 1: Sync Python dependencies without building loom-dataflow
#
# Keep this separate from the native extension build below. If that build
# fails, the resolved Python environment remains installed and can be reused
# on the next run. --inexact also preserves a previously working dataflow
# installation until its replacement has built successfully.
# ---------------------------------------------------------------------------
UV_SYNC_ARGS=(sync --project "$REPO_ROOT" --locked --inexact)
INSTALL_DATAFLOW=0
if [ "$LOOM_CAN_BUILD_DATAFLOW" = "1" ] && [ "$SKIP_DATAFLOW" = "0" ]; then
    INSTALL_DATAFLOW=1
    UV_SYNC_ARGS+=(--extra dataflow --no-install-package loom-dataflow)
else
    echo "[SKIP] loom-dataflow Python package (missing dependencies or --skip-dataflow)"
fi

if [ "$SKIP_HELION" = "0" ]; then
    UV_SYNC_ARGS+=(--extra helion)
else
    echo "[SKIP] helion-mlir Python package (--skip-helion)"
fi

echo ""
echo "=== Stage 1/2: Syncing Python environment (excluding loom-dataflow build) ==="
uv "${UV_SYNC_ARGS[@]}"

# ---------------------------------------------------------------------------
# Stage 2: Build native components
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 2/2: Building native components ==="

if [ "$INSTALL_DATAFLOW" = "1" ]; then
    bash "$REPO_ROOT/scripts/build-llvm.sh"

    DATAFLOW_SYNC_ARGS=(
        sync
        --project "$REPO_ROOT"
        --locked
        --inexact
        --extra dataflow
    )
    if [ "$REBUILD_DATAFLOW" = "1" ]; then
        DATAFLOW_SYNC_ARGS+=(--reinstall-package loom-dataflow)
    fi
    if [ "$SKIP_HELION" = "0" ]; then
        DATAFLOW_SYNC_ARGS+=(--extra helion)
    fi

    echo ""
    echo "--- Building and installing loom-dataflow ---"
    uv "${DATAFLOW_SYNC_ARGS[@]}"
fi

# ---------------------------------------------------------------------------
# Stage 2 continued: loom-mlar (Rust evaluator binary)
# ---------------------------------------------------------------------------
if [ "$LOOM_HAS_CARGO" = "1" ] && [ "$SKIP_MLAR" = "0" ]; then
    if [ -n "${LOOM_EVAL_SYSTEM:-}" ] && [ -x "${LOOM_EVAL_SYSTEM}" ]; then
        echo ""
        echo "[SKIP] loom-mlar build (using pre-built binary: $LOOM_EVAL_SYSTEM)"
    else
        echo ""
        echo "=== Building loom-mlar eval_system binary ==="
        bash "$REPO_ROOT/scripts/build-mlar.sh"
    fi
else
    echo ""
    echo "[SKIP] loom-mlar eval_system binary (Rust not available or --skip-mlar)"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "  Loom monorepo install complete"
echo "============================================"

_check_import() {
    if uv run --project "$REPO_ROOT" --no-sync python -c "import $1" 2>/dev/null; then
        printf "  %-16s %s\n" "$1" "OK"
    else
        printf "  %-16s %s\n" "$1" "NOT INSTALLED"
    fi
}

_check_import loom_pipeline
_check_import helion_mlir
_check_import loom

# Check eval_system binary
_eval_bin="${LOOM_EVAL_SYSTEM:-$REPO_ROOT/third_party/loom-mlar/tests/2d_mesh/bin/eval_system}"
if [ -x "$_eval_bin" ]; then
    printf "  %-16s %s\n" "eval_system" "OK ($_eval_bin)"
else
    printf "  %-16s %s\n" "eval_system" "NOT BUILT"
fi

echo ""
