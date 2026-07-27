#!/usr/bin/env bash
# Install the Loom workspace inside the supported Docker development image.
#
# Usage:
#   bash install-docker.sh [OPTIONS]
#
# Options:
#   --skip-mlar         Skip the loom-mlar Rust evaluator
#   --skip-dataflow     Skip loom-dataflow and loom2ttkernel
#   --skip-helion       Skip helion-mlir
#   --skip-ttkernel     Skip the optional loom2ttkernel backend
#   --rebuild-dataflow  Force reinstalling the loom-dataflow Python extension
#   --help              Show this help
#
# Environment:
#   LOOM_EVAL_SYSTEM    Path to a prebuilt eval_system binary

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SKIP_MLAR=0
SKIP_DATAFLOW=0
SKIP_HELION=0
SKIP_TTKERNEL=0
REBUILD_DATAFLOW=0

for arg in "$@"; do
    case "$arg" in
        --skip-mlar)        SKIP_MLAR=1 ;;
        --skip-dataflow)    SKIP_DATAFLOW=1 ;;
        --skip-helion)      SKIP_HELION=1 ;;
        --skip-ttkernel)    SKIP_TTKERNEL=1 ;;
        --rebuild-dataflow) REBUILD_DATAFLOW=1 ;;
        --help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $arg (see --help)" >&2
            exit 1
            ;;
    esac
done

if [ "$SKIP_DATAFLOW" = "1" ]; then
    SKIP_TTKERNEL=1
fi

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

require_file() {
    [ -f "$1" ] || fail "required file not found: $1"
}

echo ""
echo "=== Checking Docker toolchain ==="
for command_name in uv cmake ninja clang++ ld.lld cargo rustc; do
    require_command "$command_name"
done

: "${MLIR_DIR:?MLIR_DIR is not set; use the loom_dev Docker image}"
: "${LLVM_DIR:?LLVM_DIR is not set; use the loom_dev Docker image}"
: "${TTMLIR_SOURCE_DIR:?TTMLIR_SOURCE_DIR is not set; use the loom_dev Docker image}"
: "${TTMLIR_BUILD_DIR:?TTMLIR_BUILD_DIR is not set; use the loom_dev Docker image}"

require_file "$MLIR_DIR/MLIRConfig.cmake"
require_file "$LLVM_DIR/LLVMConfig.cmake"
require_file "$TTMLIR_BUILD_DIR/bin/ttmlir-opt"
require_file "$TTMLIR_BUILD_DIR/lib/libMLIRTTKernelDialect.a"
require_file "$TTMLIR_BUILD_DIR/lib/libMLIRTTMetalDialect.a"

printf "  %-18s %s\n" "uv" "$(uv --version)"
printf "  %-18s %s\n" "MLIR_DIR" "$MLIR_DIR"
printf "  %-18s %s\n" "TTMLIR_BUILD_DIR" "$TTMLIR_BUILD_DIR"

# The image keeps tt-mlir's Python venv active for compiler tools. Loom uses
# its own .venv, so hide VIRTUAL_ENV from uv and copy from caches across mount
# boundaries instead of attempting unsupported hardlinks. Keep uv's managed
# Python and cache beside the checkout so they persist with the Docker volume.
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$REPO_ROOT/.uv-python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.uv-cache}"
run_uv() {
    env -u VIRTUAL_ENV uv "$@"
}

echo ""
echo "=== Initializing Git submodules ==="
if [ -f "$REPO_ROOT/third_party/loom-dataflow/CMakeLists.txt" ] \
    && [ -f "$REPO_ROOT/third_party/helion-mlir/pyproject.toml" ] \
    && [ -f "$REPO_ROOT/third_party/loom-mlar/Cargo.toml" ] \
    && [ -f "$REPO_ROOT/third_party/loom2ttkernel/CMakeLists.txt" ]; then
    echo "  [OK] Submodules are already populated"
else
    git -c safe.directory="$REPO_ROOT" -C "$REPO_ROOT" \
        submodule update --init --recursive
fi

UV_BASE_ARGS=(
    sync
    --project "$REPO_ROOT"
    --locked
    --inexact
)

if [ "$SKIP_HELION" = "0" ]; then
    UV_BASE_ARGS+=(--extra helion)
fi

echo ""
echo "=== Syncing the Loom Python environment ==="
if [ "$SKIP_DATAFLOW" = "0" ]; then
    run_uv "${UV_BASE_ARGS[@]}" --extra dataflow --no-install-package loom-dataflow
else
    run_uv "${UV_BASE_ARGS[@]}"
fi

if [ "$SKIP_DATAFLOW" = "0" ]; then
    echo ""
    echo "=== Building loom-dataflow CMake artifacts ==="
    bash "$REPO_ROOT/third_party/loom-dataflow/build.sh" \
        --mlir-dir="$MLIR_DIR"

    echo ""
    echo "=== Installing the loom-dataflow Python extension ==="
    DATAFLOW_SYNC_ARGS=("${UV_BASE_ARGS[@]}" --extra dataflow)
    if [ "$REBUILD_DATAFLOW" = "1" ]; then
        DATAFLOW_SYNC_ARGS+=(--reinstall-package loom-dataflow)
    fi
    run_uv "${DATAFLOW_SYNC_ARGS[@]}"
fi

if [ "$SKIP_MLAR" = "0" ]; then
    if [ -n "${LOOM_EVAL_SYSTEM:-}" ] && [ -x "${LOOM_EVAL_SYSTEM}" ]; then
        echo ""
        echo "[SKIP] loom-mlar (using $LOOM_EVAL_SYSTEM)"
    else
        echo ""
        echo "=== Building loom-mlar ==="
        bash "$REPO_ROOT/scripts/build-mlar.sh"
    fi
fi

if [ "$SKIP_TTKERNEL" = "0" ]; then
    echo ""
    echo "=== Building loom2ttkernel ==="
    bash "$REPO_ROOT/third_party/loom2ttkernel/build.sh" \
        -DMLIR_DIR="$MLIR_DIR" \
        -DLLVM_DIR="$LLVM_DIR" \
        -DTTMLIR_SOURCE_DIR="$TTMLIR_SOURCE_DIR" \
        -DTTMLIR_BUILD_DIR="$TTMLIR_BUILD_DIR"
fi

echo ""
echo "============================================"
echo "  Loom Docker workspace install complete"
echo "============================================"

check_import() {
    if run_uv run --project "$REPO_ROOT" --no-sync python -c "import $1" 2>/dev/null; then
        printf "  %-18s %s\n" "$1" "OK"
    else
        printf "  %-18s %s\n" "$1" "NOT INSTALLED"
    fi
}

check_import loom
if [ "$SKIP_HELION" = "0" ]; then
    check_import helion_mlir
fi
if [ "$SKIP_DATAFLOW" = "0" ]; then
    check_import loom_pipeline
fi

EVAL_BIN="${LOOM_EVAL_SYSTEM:-$REPO_ROOT/third_party/loom-mlar/tests/2d_mesh/bin/eval_system}"
if [ "$SKIP_MLAR" = "0" ]; then
    [ -x "$EVAL_BIN" ] || fail "loom-mlar evaluator was not produced"
    printf "  %-18s %s\n" "eval_system" "OK"
fi

TTKERNEL_BIN="$REPO_ROOT/third_party/loom2ttkernel/build/bin/tileloom_to_ttkernel_opt"
if [ "$SKIP_TTKERNEL" = "0" ]; then
    [ -x "$TTKERNEL_BIN" ] || fail "loom2ttkernel executable was not produced"
    printf "  %-18s %s\n" "loom2ttkernel" "OK"
fi

echo ""
