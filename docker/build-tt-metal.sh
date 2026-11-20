#!/usr/bin/env bash
set -euo pipefail

: "${TT_METAL_HOME:?TT_METAL_HOME must point to the pinned checkout}"
cd "$TT_METAL_HOME"
git submodule update --init --recursive -- \
    tt_metal/third_party/tracy \
    tt_metal/third_party/tt_llk \
    tt_metal/third_party/umd

# Do not reuse the compiler toolchain's Python: Loom requires Python 3.10.
uv venv --python 3.10 /opt/tt-metal-venv
export PATH="/opt/tt-metal-venv/bin:/opt/ttmlir-toolchain/venv/bin:$PATH"
export VIRTUAL_ENV=/opt/tt-metal-venv
export LD_LIBRARY_PATH="/opt/openmpi-v5.0.7-ulfm/lib:${LD_LIBRARY_PATH:-}"

# Use the CI image's Clang 20 explicitly; upstream defaults to Clang 17.
./build_metal.sh --release --build-static-libs \
    --cxx-compiler-path /usr/bin/clang++-20 \
    --c-compiler-path /usr/bin/clang-20

test -x runtime/sfpi/compiler/bin/riscv32-tt-elf-g++
runtime/sfpi/compiler/bin/riscv32-tt-elf-g++ --version
test -f build/lib/_ttnn.so
# This revision builds the shared C++ API but omits it from CMake's install
# rules. setup.py expects it in build/lib when assembling the wheel.
install -m 755 build/ttnn/_ttnncpp.so build/lib/_ttnncpp.so
# Upstream setup.py selects lib64 on hosts with /usr/lib64, while the build
# installs into lib. Provide that packaging alias without moving the libraries.
if [ -d /usr/lib64 ] && [ ! -e build/lib64 ]; then
    ln -s lib build/lib64
fi

# The shallow checkout has no release tags. Give this private wheel a version
# that records its exact source commit instead of guessing a release number.
export SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0+git.$(git rev-parse HEAD)"
TT_FROM_PRECOMPILED_DIR="$TT_METAL_HOME" \
    uv build --wheel --python /opt/tt-metal-venv/bin/python \
    --out-dir /opt/loom/wheels
uv pip install --python /opt/tt-metal-venv/bin/python /opt/loom/wheels/ttnn-*.whl

# setuptools stages another copy of the wheel contents here during packaging.
rm -rf build/lib.linux-x86_64-cpython-310
