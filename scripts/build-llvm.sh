#!/bin/bash
# Configure and incrementally build the bundled LLVM/MLIR toolchain.
#
# Source:
#   third_party/llvm-project
#   commit 4efe170d858eb54432f520abb4e7f0086236748b
#
# Output:
#   build/llvm-4efe170d
#
# Environment:
#   LOOM_LLVM_JOBS  Override parallel build jobs (default: min(nproc, 32))

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLVM_SOURCE_DIR="$REPO_ROOT/third_party/llvm-project"
LLVM_BUILD_DIR="$REPO_ROOT/build/llvm-4efe170d"
LLVM_EXPECTED_REV="4efe170d858eb54432f520abb4e7f0086236748b"
LLVM_EXPECTED_VERSION="22.0.0git"
PYTHON_EXECUTABLE="$REPO_ROOT/.venv/bin/python"

if [ ! -f "$LLVM_SOURCE_DIR/llvm/CMakeLists.txt" ]; then
    echo "ERROR: llvm-project submodule is not initialized."
    echo "       Run: git submodule update --init --recursive"
    exit 1
fi

actual_rev=$(git -C "$LLVM_SOURCE_DIR" rev-parse HEAD)
if [ "$actual_rev" != "$LLVM_EXPECTED_REV" ]; then
    echo "ERROR: llvm-project is not at the required pinned revision."
    echo "       Expected: $LLVM_EXPECTED_REV"
    echo "       Found:    $actual_rev"
    echo "       Run: git submodule update --init --recursive"
    exit 1
fi

if [ ! -x "$PYTHON_EXECUTABLE" ]; then
    echo "ERROR: uv environment not found at $PYTHON_EXECUTABLE"
    echo "       Run the Python dependency stage before building LLVM."
    exit 1
fi

if [ -n "${LOOM_LLVM_JOBS:-}" ]; then
    LLVM_JOBS="$LOOM_LLVM_JOBS"
else
    detected_jobs=$(nproc)
    if [ "$detected_jobs" -gt 32 ]; then
        LLVM_JOBS=32
    else
        LLVM_JOBS="$detected_jobs"
    fi
fi

echo ""
echo "=== Configuring bundled LLVM/MLIR at 4efe170d85 ==="
echo "  Source: $LLVM_SOURCE_DIR"
echo "  Commit: $LLVM_EXPECTED_REV"
echo "  Build:  $LLVM_BUILD_DIR"
echo "  Jobs:   $LLVM_JOBS"

cmake \
    -S "$LLVM_SOURCE_DIR/llvm" \
    -B "$LLVM_BUILD_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DPython3_EXECUTABLE="$PYTHON_EXECUTABLE" \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD=Native \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DMLIR_INCLUDE_TESTS=OFF \
    -DLLVM_USE_LINKER=lld

echo ""
echo "=== Building bundled LLVM/MLIR at 4efe170d85 ==="
cmake --build "$LLVM_BUILD_DIR" \
    --target mlir-opt mlir-lsp-server \
    --parallel "$LLVM_JOBS"

MLIR_CONFIG="$LLVM_BUILD_DIR/lib/cmake/mlir/MLIRConfig.cmake"
LLVM_VERSION_FILE="$LLVM_BUILD_DIR/lib/cmake/llvm/LLVMConfigVersion.cmake"

if [ ! -f "$MLIR_CONFIG" ] || [ ! -f "$LLVM_VERSION_FILE" ]; then
    echo "ERROR: LLVM/MLIR build completed without the expected CMake packages."
    exit 1
fi

llvm_version=$(sed -n 's/set(PACKAGE_VERSION "\([^"]*\)").*/\1/p' "$LLVM_VERSION_FILE" | head -1)
if [ "$llvm_version" != "$LLVM_EXPECTED_VERSION" ]; then
    echo "ERROR: Expected LLVM $LLVM_EXPECTED_VERSION, but the build reports $llvm_version."
    exit 1
fi

echo ""
echo "Bundled LLVM/MLIR is ready:"
echo "  Version:  $llvm_version"
echo "  MLIR_DIR: $LLVM_BUILD_DIR/lib/cmake/mlir"
