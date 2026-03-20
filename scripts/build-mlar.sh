#!/bin/bash
# Build the loom-mlar eval_core evaluator binary.
#
# The architecture definition currently lives in test code, so we invoke
# `cargo test` to trigger the binary generation.  A future refactor should
# move the architecture to src/ and create a proper [[bin]] target.
#
# Usage:
#   bash scripts/build-mlar.sh
#
# Output:
#   third_party/loom-mlar/bin/eval_core

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

echo "Building eval_core binary (this may take a while on first run)..."
cd "$MLAR_DIR"

# Run the specific test that generates the evaluator binary.
cargo test --test 2d_mesh test_generate_core_evaluator_binary --release -- --nocapture

# Verify the binary was produced.
GENERATED="$MLAR_DIR/tests/2d_mesh/evaluators/eval_core"
if [ ! -x "$GENERATED" ]; then
    echo "ERROR: eval_core binary was not generated at $GENERATED"
    exit 1
fi

# Copy to canonical location.
mkdir -p "$MLAR_DIR/bin"
cp "$GENERATED" "$MLAR_DIR/bin/eval_core"
chmod +x "$MLAR_DIR/bin/eval_core"

echo ""
echo "eval_core binary built successfully:"
echo "  $MLAR_DIR/bin/eval_core"
