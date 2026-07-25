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
#   SKIP_MLAR      - set to 1 to skip cargo/rustc checks

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
        && [ -f "$repo_root/third_party/loom-mlar/Cargo.toml" ] \
        && [ -f "$repo_root/third_party/llvm-project/llvm/CMakeLists.txt" ]; then
        _ok "Git submodules populated"
    else
        _fail "Git submodules not initialized"
        _hint "Run: git submodule update --init --recursive"
        (( _PREFLIGHT_ERRORS++ ))
    fi
}

check_uv() {
    if command -v uv &>/dev/null; then
        local ver
        ver=$(uv --version)
        _ok "$ver"
    else
        _fail "uv not found"
        _hint "Install uv: https://docs.astral.sh/uv/getting-started/installation/"
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
        if [ "$major" -gt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -ge 20 ]; }; then
            _ok "cmake $ver"
        else
            _fail "cmake >= 3.20 required (found $ver)"
            _hint "Install: apt install cmake"
            LOOM_CAN_BUILD_DATAFLOW=0
            (( _PREFLIGHT_ERRORS++ ))
        fi
    else
        _fail "cmake not found"
        _hint "Install: apt install cmake"
        LOOM_CAN_BUILD_DATAFLOW=0
        (( _PREFLIGHT_ERRORS++ ))
    fi
}

check_ninja() {
    if command -v ninja &>/dev/null; then
        _ok "ninja"
    else
        _fail "ninja not found"
        _hint "Install: apt install ninja-build"
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

check_llvm_submodule() {
    local expected_version="22.0.0git"
    local expected_rev="4efe170d858eb54432f520abb4e7f0086236748b"
    local expected_short_rev="${expected_rev:0:10}"
    local llvm_source="${REPO_ROOT:-.}/third_party/llvm-project"
    local actual_rev
    actual_rev=$(git -C "$llvm_source" rev-parse HEAD 2>/dev/null || true)
    if [ "$actual_rev" = "$expected_rev" ]; then
        _ok "Bundled LLVM/MLIR $expected_version source ($expected_short_rev)"
    else
        _fail "Bundled llvm-project is not at pinned commit $expected_rev"
        _hint "Run: git submodule update --init --recursive"
        LOOM_CAN_BUILD_DATAFLOW=0
        (( _PREFLIGHT_ERRORS++ ))
    fi
}

check_cargo() {
    if command -v cargo &>/dev/null && command -v rustc &>/dev/null; then
        local ver
        ver=$(rustc --version | grep -oP '[0-9]+\.[0-9]+\.[0-9]+')
        _ok "Rust toolchain (rustc $ver)"
    else
        _fail "cargo/rustc not found"
        _hint "Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        _hint "Then restart your shell and re-run this script."
        LOOM_HAS_CARGO=0
        (( _PREFLIGHT_ERRORS++ ))
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
    check_uv

    if [ "${SKIP_DATAFLOW:-0}" != "1" ]; then
        check_cmake
        check_ninja
        check_lld
        check_cxx_compiler
        check_llvm_submodule
    else
        _warn "Skipping loom-dataflow checks (--skip-dataflow)"
        LOOM_CAN_BUILD_DATAFLOW=0
        (( _PREFLIGHT_WARNINGS++ ))
    fi

    if [ "${SKIP_MLAR:-0}" != "1" ]; then
        check_cargo
    else
        _warn "Skipping Rust checks (--skip-mlar)"
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
