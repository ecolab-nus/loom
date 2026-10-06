#!/bin/bash
# Build the loom-mlar eval_system evaluator binaries.
#
# Each architecture lives in third_party/loom-mlar/tests/<arch>/ (arch.rs plus
# processors/*.mlir and *.perf.yaml).  The architecture definition currently
# lives in test code, so we invoke `cargo test --test <arch>` to export the
# hardware spec and generate the evaluator binary.
#
# Usage:
#   bash scripts/build-mlar.sh                 # all architectures
#   bash scripts/build-mlar.sh blackhole       # selected architectures
#
# Output (per arch):
#   third_party/loom-mlar/tests/<arch>/2d_mesh_torus.mlir   (hw_spec)
#   third_party/loom-mlar/tests/<arch>/bin/eval_system

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MLAR_DIR="$REPO_ROOT/third_party/loom-mlar"

if [ ! -f "$MLAR_DIR/Cargo.toml" ]; then
    echo "ERROR: loom-mlar not found at $MLAR_DIR"
    echo "       Run: git submodule update --init --recursive"
    exit 1
fi

if ! command -v cargo &>/dev/null; then
    echo "ERROR: cargo not found. Install Rust: https://rustup.rs"
    exit 1
fi

ARCHS=("$@")
if [ ${#ARCHS[@]} -eq 0 ]; then
    ARCHS=(wormhole blackhole)
fi

cd "$MLAR_DIR"

for ARCH in "${ARCHS[@]}"; do
    if [ ! -f "$MLAR_DIR/tests/$ARCH/main.rs" ]; then
        echo "ERROR: unknown architecture '$ARCH' (no tests/$ARCH/main.rs)"
        exit 1
    fi

    echo "Building $ARCH hw_spec + eval_system (this may take a while on first run)..."
    cargo test --test "$ARCH" --release -- --nocapture \
        test_export_2d_mesh_torus_mlir test_generate_system_evaluator_binary

    GENERATED="$MLAR_DIR/tests/$ARCH/bin/eval_system"
    if [ ! -x "$GENERATED" ]; then
        echo "ERROR: eval_system binary was not generated at $GENERATED"
        exit 1
    fi

    echo ""
    echo "$ARCH eval_system built successfully:"
    echo "  $GENERATED"
done
