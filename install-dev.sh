#!/bin/bash
# One-click developer install for the Loom monorepo.
#
# Installs the core pipeline subprojects in editable mode in the correct order:
#   1. Git submodule init/update
#   2. Pre-flight dependency checks
#   3. loom-dataflow  (C++ MLIR pipeline, triggers CMake build)
#   4. helion-mlir    (Python, Helion-to-MLIR lowering)
#   5. loom-mlar      (Rust evaluator binary)
#   6. loom           (root orchestrator meta-package)
#
# Usage:
#   bash install-dev.sh [OPTIONS]
#
# Options:
#   --mlir-dir=PATH     Path to MLIR cmake config
#                        (default: $MLIR_DIR or /opt/llvm-mlir/lib/cmake/mlir)
#   --skip-mlar         Skip building the loom-mlar Rust evaluator
#   --skip-dataflow     Skip building loom-dataflow (C++ MLIR passes)
#   --skip-helion       Skip installing helion-mlir
#   --help              Show this help message
#
# Environment:
#   MLIR_DIR            Path to MLIR cmake config directory
#   LOOM_EVAL_SYSTEM    Path to pre-built eval_system binary (skips Rust build)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SKIP_MLAR=0
SKIP_DATAFLOW=0
SKIP_HELION=0

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --mlir-dir=*)    export MLIR_DIR="${arg#*=}" ;;
        --skip-mlar)     SKIP_MLAR=1 ;;
        --skip-dataflow) SKIP_DATAFLOW=1 ;;
        --skip-helion)   SKIP_HELION=1 ;;
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

export MLIR_DIR="${MLIR_DIR:-/opt/llvm-mlir/lib/cmake/mlir}"
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
# Step 2: Auto-install missing Python build dependencies
# ---------------------------------------------------------------------------
if [ -n "${LOOM_NEED_BUILD_DEPS:-}" ]; then
    echo ""
    echo "=== Installing missing Python build deps: $LOOM_NEED_BUILD_DEPS ==="
    pip install $LOOM_NEED_BUILD_DEPS
fi

# ---------------------------------------------------------------------------
# Step 3: loom-dataflow (C++ MLIR pipeline)
# ---------------------------------------------------------------------------
if [ "$LOOM_CAN_BUILD_DATAFLOW" = "1" ] && [ "$SKIP_DATAFLOW" = "0" ]; then
    echo ""
    echo "=== Installing loom-dataflow (editable, triggers CMake build) ==="
    pip install -e "$REPO_ROOT/third_party/loom-dataflow" -v --no-build-isolation
else
    echo ""
    echo "[SKIP] loom-dataflow (missing dependencies or --skip-dataflow)"
fi

# ---------------------------------------------------------------------------
# Step 4: helion-mlir (Python)
# ---------------------------------------------------------------------------
if [ "$SKIP_HELION" = "0" ]; then
    echo ""
    echo "=== Installing helion-mlir (editable) ==="
    # torch-mlir dev wheels are not on PyPI; use the find-links URL from
    # helion-mlir/requirements.txt so pip can locate the package.
    pip install -e "$REPO_ROOT/third_party/helion-mlir" \
        -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
else
    echo ""
    echo "[SKIP] helion-mlir (--skip-helion)"
fi

# ---------------------------------------------------------------------------
# Step 5: loom-mlar (Rust evaluator binary)
# ---------------------------------------------------------------------------
if [ "$LOOM_HAS_CARGO" = "1" ] && [ "$SKIP_MLAR" = "0" ]; then
    if [ -n "${LOOM_EVAL_CORE:-}" ] && [ -x "${LOOM_EVAL_CORE}" ]; then
        echo ""
        echo "[SKIP] loom-mlar build (using pre-built binary: $LOOM_EVAL_CORE)"
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
# Step 6: Root loom package
# ---------------------------------------------------------------------------
echo ""
echo "=== Installing loom (editable) ==="
pip install -e "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "  Loom monorepo install complete"
echo "============================================"

_check_import() {
    if python3 -c "import $1" 2>/dev/null; then
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
