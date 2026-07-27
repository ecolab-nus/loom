# Docker Development

The recommended environment for developing the Tenstorrent backend is the
prebuilt [`ftod/loom_dev:latest`](https://hub.docker.com/r/ftod/loom_dev)
image on Docker Hub. It is built from the repository
[`Dockerfile`](https://github.com/ecolab-nus/loom/blob/main/Dockerfile).

## Image Contents

The image contains:

- the LLVM/MLIR 22 toolchain from the Tenstorrent tt-mlir CI image;
- tt-mlir at commit
  `5009f4764a31ff08e7cd838ed5747ae8a368a7e6`;
- TT-Metal source at commit
  `ad07818dd4d704d654359c8392794ef8ea8ceec4`;
- tt-mlir compiler binaries and static libraries required by
  `loom2ttkernel`;
- Rust installed through rustup;
- `tt-smi` and `tt-flash`.

The important paths are exported automatically:

```text
MLIR_DIR=/opt/ttmlir-toolchain/lib/cmake/mlir
LLVM_DIR=/opt/ttmlir-toolchain/lib/cmake/llvm
TTMLIR_SOURCE_DIR=/opt/tt-mlir
TTMLIR_BUILD_DIR=/opt/tt-mlir/build
TT_METAL_HOME=/opt/tt-mlir/third_party/tt-metal/src/tt-metal
```

## 1. Prerequisites and Source Checkout

Docker Engine is required. Buildx is only required when rebuilding the image
locally:

```bash
docker version
docker buildx version  # optional for Docker Hub users
```

Clone Loom together with all submodules:

```bash
git clone --recurse-submodules https://github.com/ecolab-nus/loom.git
cd loom
```

For an existing checkout:

```bash
git submodule update --init --recursive
```

## 2. Download or Build the Image

Download the prebuilt image:

```bash
docker pull ftod/loom_dev:latest
docker image inspect ftod/loom_dev:latest >/dev/null
```

The published image currently targets `linux/amd64`. It is ready to use after
the pull; no local image build is necessary.

To rebuild after changing the Dockerfile, pull the Tenstorrent base and assign
the result a separate local tag:

```bash
docker pull ghcr.io/tenstorrent/tt-mlir/tt-mlir-ci-ubuntu-24-04:latest

docker buildx build \
  --load \
  --tag loom_dev:local \
  .
```

The first build downloads and compiles tt-mlir and can take several minutes.
Subsequent builds reuse the BuildKit cache. Expect roughly 16 GB of local disk
usage for the expanded image, in addition to the compilation cache.

Override pinned source revisions only when deliberately testing another
combination:

```bash
docker buildx build \
  --load \
  --tag loom_dev:local \
  --build-arg TTMLIR_COMMIT=<tt-mlir-commit> \
  --build-arg TT_METAL_COMMIT=<tt-metal-commit> \
  .
```

For an offline machine, transfer an image archive and load it:

```bash
gzip -dc loom_dev-latest.tar.gz | docker load
docker image inspect ftod/loom_dev:latest >/dev/null
```

Create the archive on a machine that already has the image:

```bash
docker save ftod/loom_dev:latest | gzip >loom_dev-latest.tar.gz
```

Use `loom_dev:local` instead of `ftod/loom_dev:latest` in the commands below
when testing a locally rebuilt image.

## 3. Verify the Image

```bash
docker run --rm ftod/loom_dev:latest bash -lc '
  set -e
  test -f "$MLIR_DIR/MLIRConfig.cmake"
  test -f "$LLVM_DIR/LLVMConfig.cmake"
  test -f "$TTMLIR_BUILD_DIR/lib/libMLIRTTKernelDialect.a"
  test -f "$TTMLIR_BUILD_DIR/lib/libMLIRTTMetalDialect.a"
  git -C "$TTMLIR_SOURCE_DIR" rev-parse HEAD
  git -C "$TT_METAL_HOME" rev-parse HEAD
  ttmlir-opt --version
  mlir-opt --version
  rustc --version
  cargo --version
'
```

The two Git revisions should match the commits listed under Image Contents.

## 4. Start a Development Container

From the repository root, create a named container and mount the checkout at
`/workspace`:

```bash
docker run -it \
  --name loom-dev \
  --mount type=bind,src="$(pwd)",dst=/workspace \
  --workdir /workspace \
  ftod/loom_dev:latest
```

Leave it with `exit` and resume the same environment later:

```bash
docker start --attach --interactive loom-dev
```

For a disposable container:

```bash
docker run --rm -it \
  --mount type=bind,src="$(pwd)",dst=/workspace \
  --workdir /workspace \
  ftod/loom_dev:latest
```

Commands run as root, so generated files in the mounted checkout may be owned
by root on the host. Restore ownership after exiting if needed:

```bash
sudo chown -R "$(id -u):$(id -g)" .venv build third_party/*/build
```

Omit paths that do not exist.

## 5. Build the Workspace

The commands in this section run inside the container. Install `uv` once in a
named container:

```bash
python -m pip install "uv>=0.11"
uv --version
```

Discard build products made with another host or toolchain:

```bash
rm -rf \
  build \
  third_party/loom-dataflow/build \
  third_party/loom2ttkernel/build \
  third_party/loom-mlar/target
```

Prepare the locked Python environment, install the `loom-dataflow` extension,
and build the MLAR evaluator:

```bash
uv sync --locked --inexact \
  --extra dataflow \
  --extra helion \
  --no-install-package loom-dataflow

uv sync --locked --inexact \
  --extra dataflow \
  --extra helion \
  --reinstall-package loom-dataflow

bash scripts/build-mlar.sh
```

Create a persistent CMake build of `loom-dataflow` to supply the static
libraries and generated dialect headers required by `loom2ttkernel`:

```bash
bash third_party/loom-dataflow/build.sh \
  --mlir-dir="$MLIR_DIR"
```

Build the optional TTKernel backend:

```bash
bash third_party/loom2ttkernel/build.sh \
  -DMLIR_DIR="$MLIR_DIR" \
  -DLLVM_DIR="$LLVM_DIR" \
  -DTTMLIR_SOURCE_DIR="$TTMLIR_SOURCE_DIR" \
  -DTTMLIR_BUILD_DIR="$TTMLIR_BUILD_DIR"
```

Do not use `install-dev.sh` for this workflow. That script is the native host
installation path and builds the repository's LLVM/MLIR toolchain. The
commands above reuse the toolchain in the image.

## 6. Smoke Tests

```bash
uv run python -c 'import loom, loom_pipeline, helion_mlir'

test -x third_party/loom-mlar/tests/2d_mesh/bin/eval_system
test -x third_party/loom2ttkernel/build/bin/tileloom_to_ttkernel_opt

uv run pytest
```

Run an example kernel:

```bash
uv run python kernels/matmul.py \
  --config kernels/config_files/matmul.json \
  --njobs 16 \
  --debug
```

See the [usage guide](usage.md) for configuration and output details.

## 7. Tenstorrent Hardware Access

LLVM/MLIR compilation does not require a Tenstorrent device. For hardware
access, the kernel driver and HugePages must be configured on the host. The
Dockerfile uses tt-installer's container mode and does not modify the host.

Verify the host:

```bash
tt-smi
test -e /dev/tenstorrent
test -d /dev/hugepages-1G
```

Pass all Tenstorrent devices and the HugePages mount into the container:

```bash
docker run -it \
  --name loom-dev-hw \
  --device /dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount type=bind,src="$(pwd)",dst=/workspace \
  --workdir /workspace \
  ftod/loom_dev:latest
```

Do not pass only an individual entry such as `/dev/tenstorrent/0`; all devices
must be passed through together. See the
[Tenstorrent Docker setup guide](https://docs.tenstorrent.com/tt-forge-onnx/getting_started_docker.html)
and
[tt-installer container-mode guidance](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/tt-installer/)
for host setup.

## 8. Common Operations

```bash
# Show the image
docker image ls ftod/loom_dev:latest

# Show stopped and running development containers
docker ps --all --filter name=loom-dev

# Resume the named container
docker start --attach --interactive loom-dev

# Delete the container; bind-mounted source remains on the host
docker rm loom-dev

# Rebuild after changing the Dockerfile
docker buildx build --load --tag loom_dev:local .
```
