#!/bin/bash
# Developer install script for the Loom monorepo.
#
# ---------------------------------------------------------------------------
# Installs all three packages in editable mode in the correct order:
#   1. loom-dataflow  (triggers CMake build for C++ pipeline)
#   2. helion-mlir    (pure Python)
#   3. loom           (root meta-package, no-deps to avoid PyPI lookup)
#
# Usage:
#   bash scripts/install-dev.sh
#
# Environment:
#   MLIR_DIR  - Path to MLIR cmake config (default: /opt/llvm-mlir/lib/cmake/mlir)

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export MLIR_DIR="${MLIR_DIR:-/opt/llvm-mlir/lib/cmake/mlir}"

# ---------------------------------------------------------------------------
# Pre-install build-time dependencies for loom-dataflow.
# --no-build-isolation skips pip's isolated build env, so scikit-build-core
# and pybind11 must be present in the active environment before the build.
# cmake and ninja are available system-wide (/usr/bin); only the Python
# build backend needs to be installed here.
# ---------------------------------------------------------------------------
echo ""
echo "=== Installing loom-dataflow (editable, triggers CMake build) ==="
pip install -e "$REPO_ROOT/third_party/loom-dataflow" -v --no-build-isolation

echo ""
echo "=== Installing helion-mlir (editable) ==="
pip install -e "$REPO_ROOT/third_party/helion-mlir"

echo ""
echo "=== Installing loom (editable) ==="
pip install -e "$REPO_ROOT"

echo ""
echo "=== All packages installed in editable mode ==="
echo "  - loom-dataflow  (import loom_pipeline)"
echo "  - helion-mlir    (import helion_mlir)"
echo "  - loom           (import loom)"
echo ""