#!/bin/bash
# Pre-flight dependency checker for the Loom monorepo.
#
# Usage:
#   source scripts/preflight.sh
#   run_preflight_checks          # run all checks
#
# After sourcing, the following variables are exported:
#   LOOM_CAN_BUILD_DATAFLOW  (0 or 1)
#   LOOM_HAS_CARGO           (0 or 1)
#
# Callers may set these variables before sourcing to influence behaviour:
#   SKIP_DATAFLOW  - set to 1 to skip cmake/MLIR checks
#   SKIP_RUST      - set to 1 to skip cargo/rustc checks
#   MLIR_DIR       - path to MLIR cmake config directory

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_PREFLIGHT_ERRORS=0
_PREFLIGHT_WARNINGS=0

LOOM_CAN_BUILD_DATAFLOW=1
LOOM_HAS_CARGO=1

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_ok()   { printf "  [OK]   %s\n" "$1"; }
_fail() { printf "  [FAIL] %s\n" "$1"; }
_warn() { printf "  [WARN] %s\n" "$1"; }
_hint() { printf "         -> %s\n" "$1"; }

# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

check_submodules() {
    local repo_root="${REPO_ROOT:-.}"
    if [ -f "$repo_root/third_party/loom-dataflow/CMakeLists.txt" ] \
        && [ -f "$repo_root/third_party/helion-mlir/pyproject.toml" ] \
        && [ -f "$repo_root/third_party/loom-mlar/Cargo.toml" ]; then
        _ok "Git submodules populated"
    else
        _fail "Git submodules not initialized"
        _hint "Run: git submodule update --init --recursive"
        (( _PREFLIGHT_ERRORS++ ))
    fi
}

check_python() {
    if command -v python3 &>/dev/null; then
        local ver
        ver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        local major minor
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            _ok "Python $ver"
        else
            _fail "Python >= 3.10 required (found $ver)"
            (( _PREFLIGHT_ERRORS++ ))
        fi
    else
        _fail "python3 not found"
        _hint "Install Python 3.10+ from https://www.python.org/downloads/"
        (( _PREFLIGHT_ERRORS++ ))
    fi
}

check_pip() {
    if python3 -m pip --version &>/dev/null; then
        _ok "pip"
    else
        _fail "pip not found"
        _hint "Install: python3 -m ensurepip --upgrade"
        (( _PREFLIGHT_ERRORS++ ))
    fi
}

check_cmake() {
    if command -v cmake &>/dev/null; then
        local ver
        ver=$(cmake --version | head -1 | grep -oP '[0-9]+\.[0-9]+')
        local major minor
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 20 ]; then
            _ok "cmake $ver"
        else
            _fail "cmake >= 3.20 required (found $ver)"
            _hint "Install: apt install cmake  OR  pip install cmake"
            LOOM_CAN_BUILD_DATAFLOW=0
            (( _PREFLIGHT_ERRORS++ ))
        fi
    else
        _fail "cmake not found"
        _hint "Install: apt install cmake  OR  pip install cmake"
        LOOM_CAN_BUILD_DATAFLOW=0
        (( _PREFLIGHT_ERRORS++ ))
    fi
}

check_ninja() {
    if command -v ninja &>/dev/null; then
        _ok "ninja"
    else
        _fail "ninja not found"
        _hint "Install: apt install ninja-build  OR  pip install ninja"
        LOOM_CAN_BUILD_DATAFLOW=0
        (( _PREFLIGHT_ERRORS++ ))
    fi
}

check_lld() {
    if command -v ld.lld &>/dev/null; then
        _ok "lld linker"
    else
        _fail "ld.lld not found"
        _hint "Install: apt install lld"
        LOOM_CAN_BUILD_DATAFLOW=0
        (( _PREFLIGHT_ERRORS++ ))
    fi
}

check_cxx_compiler() {
    local found=0
    for cc in g++ clang++; do
        if command -v "$cc" &>/dev/null; then
            _ok "C++ compiler ($cc)"
            found=1
            break
        fi
    done
    if [ "$found" -eq 0 ]; then
        _fail "No C++17 compiler found (g++ or clang++)"
        _hint "Install: apt install build-essential"
        LOOM_CAN_BUILD_DATAFLOW=0
        (( _PREFLIGHT_ERRORS++ ))
    fi
}

check_mlir_dir() {
    local mlir_dir="${MLIR_DIR:-/opt/llvm-mlir/lib/cmake/mlir}"
    if [ -f "$mlir_dir/MLIRConfig.cmake" ]; then
        _ok "MLIR_DIR ($mlir_dir)"
    else
        _fail "MLIRConfig.cmake not found in MLIR_DIR=$mlir_dir"
        _hint "Set MLIR_DIR to your MLIR cmake config directory"
        _hint "  e.g. export MLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir"
        LOOM_CAN_BUILD_DATAFLOW=0
        (( _PREFLIGHT_ERRORS++ ))
    fi
}

check_python_build_deps() {
    local missing=()
    python3 -c "import scikit_build_core" &>/dev/null || missing+=("scikit-build-core")
    python3 -c "import pybind11" &>/dev/null || missing+=("pybind11")

    if [ ${#missing[@]} -eq 0 ]; then
        _ok "Python build deps (scikit-build-core, pybind11)"
    else
        _warn "Missing Python build deps: ${missing[*]}"
        _hint "Will auto-install: pip install ${missing[*]}"
        LOOM_NEED_BUILD_DEPS="${missing[*]}"
        (( _PREFLIGHT_WARNINGS++ ))
    fi
}

check_cargo() {
    if command -v cargo &>/dev/null && command -v rustc &>/dev/null; then
        local ver
        ver=$(rustc --version | grep -oP '[0-9]+\.[0-9]+\.[0-9]+')
        _ok "Rust toolchain (rustc $ver)"
    else
        _warn "cargo/rustc not found — loom-mlar eval_core binary will not be built"
        _hint "Install: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        LOOM_HAS_CARGO=0
        (( _PREFLIGHT_WARNINGS++ ))
    fi
}

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
run_preflight_checks() {
    _PREFLIGHT_ERRORS=0
    _PREFLIGHT_WARNINGS=0
    LOOM_CAN_BUILD_DATAFLOW=1
    LOOM_HAS_CARGO=1

    check_submodules
    check_python
    check_pip

    if [ "${SKIP_DATAFLOW:-0}" != "1" ]; then
        check_cmake
        check_ninja
        check_lld
        check_cxx_compiler
        check_mlir_dir
        check_python_build_deps
    else
        _warn "Skipping loom-dataflow checks (--skip-dataflow)"
        LOOM_CAN_BUILD_DATAFLOW=0
        (( _PREFLIGHT_WARNINGS++ ))
    fi

    if [ "${SKIP_RUST:-0}" != "1" ]; then
        check_cargo
    else
        _warn "Skipping Rust checks (--skip-rust)"
        LOOM_HAS_CARGO=0
        (( _PREFLIGHT_WARNINGS++ ))
    fi

    echo ""
    if [ "$_PREFLIGHT_ERRORS" -gt 0 ]; then
        echo "Pre-flight: $_PREFLIGHT_ERRORS error(s), $_PREFLIGHT_WARNINGS warning(s)"
        echo "Fix the errors above before proceeding."
        return 1
    else
        echo "Pre-flight: OK ($_PREFLIGHT_WARNINGS warning(s))"
        return 0
    fi
}
