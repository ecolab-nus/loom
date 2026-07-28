#!/usr/bin/env bash
# Install the Loom workspace inside the supported Docker development image.
#
# Usage:
#   bash install-docker.sh [OPTIONS]
#
# Options:
#   --clean             Remove generated files before installing
#   --clean-only        Remove generated files and exit
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
CLEAN=0
CLEAN_ONLY=0
SKIP_MLAR=0
SKIP_DATAFLOW=0
SKIP_HELION=0
SKIP_TTKERNEL=0
REBUILD_DATAFLOW=0

for arg in "$@"; do
    case "$arg" in
        --clean)             CLEAN=1 ;;
        --clean-only)        CLEAN=1; CLEAN_ONLY=1 ;;
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

remove_generated_path() {
    local generated_path="$1"
    local relative_path

    case "$generated_path" in
        "$REPO_ROOT"/*) ;;
        *) fail "refusing to clean path outside the repository: $generated_path" ;;
    esac

    if [ ! -e "$generated_path" ] && [ ! -L "$generated_path" ]; then
        return
    fi

    relative_path="${generated_path#"$REPO_ROOT"/}"
    echo "  remove $relative_path"
    rm -rf -- "$generated_path"
}

remove_if_ignored() {
    local generated_path="$1"
    local git_root
    local relative_path

    git_root="$(
        git -C "$(dirname "$generated_path")" \
            rev-parse --show-toplevel 2>/dev/null
    )" || return

    case "$git_root" in
        "$REPO_ROOT"|"$REPO_ROOT"/*) ;;
        *) fail "refusing to clean path owned by a repository outside Loom" ;;
    esac

    relative_path="${generated_path#"$git_root"/}"
    if git -C "$git_root" ls-files --error-unmatch \
        -- "$relative_path" >/dev/null 2>&1; then
        return
    fi

    if git -C "$git_root" check-ignore -q -- "$relative_path"; then
        remove_generated_path "$generated_path"
    fi
}

GENERATED_DIRECTORY_NAMES=(
    build target debug dist
    .venv .uv-cache .uv-python
    __pycache__ .pytest_cache .ruff_cache .mypy_cache
    .cache .triton_cache .tox .nox htmlcov .hypothesis
    .eggs '*.egg-info' CMakeFiles Testing _deps
    .docusaurus node_modules tmp_logs tmp_output
)

GENERATED_FILE_NAMES=(
    '*.pyc' '*.pyo'
    '*.o' '*.obj' '*.so' '*.dylib' '*.dll' '*.a' '*.lib' '*.d'
    '*.gch' '*.pch'
    '*.tmp' '*.log' '*.swp' '*.swo' '*~' temp.json
    CMakeCache.txt cmake_install.cmake compile_commands.json
)

find_generated_paths() {
    local path_kind="$1"
    shift
    local candidate_name
    local -a name_expression=()

    for candidate_name in "$@"; do
        if [ "${#name_expression[@]}" -gt 0 ]; then
            name_expression+=(-o)
        fi
        name_expression+=(-name "$candidate_name")
    done

    if [ "$path_kind" = "directory" ]; then
        find "$REPO_ROOT" -xdev \
            \( -name .git -o -path "$REPO_ROOT/.codex" \) -prune -o \
            -type d \( "${name_expression[@]}" \) -prune -print0
    else
        find "$REPO_ROOT" -xdev \
            \( -name .git -o -path "$REPO_ROOT/.codex" \) -prune -o \
            \( -type f -o -type l \) \
            \( "${name_expression[@]}" \) -print0
    fi
}

clean_generated_paths() {
    local path_kind="$1"
    shift
    local generated_path

    while IFS= read -r -d '' generated_path; do
        remove_if_ignored "$generated_path"
    done < <(find_generated_paths "$path_kind" "$@")
}

clean_ignored_children() {
    local generated_directory="$1"
    local generated_path

    if [ ! -d "$generated_directory" ]; then
        return
    fi

    while IFS= read -r -d '' generated_path; do
        remove_if_ignored "$generated_path"
    done < <(
        find "$generated_directory" -mindepth 1 -maxdepth 1 -print0
    )
}

clean_workspace() {
    local cleanup_command
    local generated_path

    for cleanup_command in dirname find git rm; do
        require_command "$cleanup_command"
    done

    echo ""
    echo "=== Cleaning generated workspace files ==="
    clean_generated_paths \
        directory "${GENERATED_DIRECTORY_NAMES[@]}"
    clean_generated_paths \
        file "${GENERATED_FILE_NAMES[@]}"
    clean_ignored_children \
        "$REPO_ROOT/third_party/loom-mlar/tests/2d_mesh/bin"

    if [ -d "$REPO_ROOT/test" ]; then
        while IFS= read -r -d '' generated_path; do
            remove_if_ignored "$generated_path"
        done < <(
            find "$REPO_ROOT/test" -mindepth 2 -maxdepth 2 \
                -type d -name constraints -print0
        )
    fi

    echo "Workspace cleanup complete"
}

if [ "$CLEAN" = "1" ]; then
    clean_workspace
    if [ "$CLEAN_ONLY" = "1" ]; then
        exit 0
    fi
fi

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
