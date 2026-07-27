<img src="assets/loom-logo.svg" alt="Loom Logo" width="250">

# Loom Monorepo

Loom is an end-to-end compilation stack for ML kernels targeting spatial hardware architectures. The project is organized around four subrepos plus a root Python package that connects them into one workflow.

## Four Subrepos

| Subrepo | Responsibility |
|---------|----------------|
| `third_party/helion-mlir` | Lowers Helion kernels into high-level MLIR with affine control flow and linalg-on-tensors compute. |
| `third_party/loom-dataflow` | Runs MLIR exploration/materialization passes, emits ETG JSON, and exposes the C++ pipeline through Python bindings. |
| `third_party/loom-mlar` | Describes hardware with MLAR and resolves ETG schedules against architecture performance models. |
| `third_party/loom2ttkernel` | Optionally lowers Loom-produced bufferized MLIR into TTKernel/tt-mlir/tt-metal style codegen inputs. |

## Root Repo Responsibilities

The root repo is the glue layer and active solver package for the stack:

```
loom-monorepo/
├── loom/                    # Python orchestration package
│   ├── pipeline.py          # End-to-end pipeline stages and output layout
│   ├── kernel_base.py       # LoomKernel base class and inherited kernel CLI
│   ├── solver/              # CPMpy/CP-SAT block-size optimizer
│   └── loom_utils/          # ETG loading, MLAR bridge, AST/modeling helpers, timers
├── kernels/                 # Example kernel entrypoints and config files
├── scripts/                 # Install, preflight, and MLAR build helpers
├── install-dev.sh           # Developer install script for the core stack
├── test/                    # Generated/integration artifacts
├── tests/                   # Python regression tests
└── third_party/             # Loom subrepos plus the pinned llvm-project toolchain
```

The root `loom` package connects the subrepos in process: it asks `helion-mlir` for stage-00 MLIR, calls `loom-dataflow` Python bindings for exploration and materialization, sends ETG variants through `loom-mlar`, and uses the active `loom.solver` CPMpy/CP-SAT optimizer to choose block-size assignments. The solver consumes resolved ETG constraints and timing expressions, searches finite symbol domains, and returns per-candidate block sizes for materialization.

## Compilation Pipeline

| Stage | Name | Component | Description |
|-------|------|-----------|-------------|
| 0 | **Helion Frontend** | `helion-mlir` | Converts a bound Helion kernel into high-level MLIR. |
| 1 | **Dataflow Exploration** | `loom-dataflow` | Explores hardware mappings, annotates reuse/copy choices, and emits explored MLIR plus ETG JSON. |
| 2 | **ETG Resolution** | `loom-mlar` | Resolves ETG variants against architecture performance models. |
| 3 | **Block-Size Solve** | `loom.solver` | Uses CPMpy/CP-SAT to find feasible, low-cost block-size assignments. |
| 4 | **Materialization** | `loom-dataflow` | Applies selected block sizes and lowers tensor IR to bufferized Loom MLIR. |
| 5 | **Optional TT Lowering** | `loom2ttkernel` | Lowers bufferized Loom MLIR toward TTKernel codegen when that backend is installed. |

When `assigned_block_size` is provided, the root pipeline bypasses Stage 3 and sends those values directly to materialization. In debug mode, Loom can still generate and resolve ETG data to produce manual latency breakdowns.

## Subrepo Notes

### `helion-mlir`

Python frontend for Helion kernels. It starts from Helion Device IR FX graphs, maps control flow to `affine.for`/`affine.parallel`, represents memory updates with tensor IR, and uses torch-mlir for ATen/linalg lowering.

### `loom-dataflow`

C++/MLIR compiler infrastructure with pybind11 bindings. It owns the ADL and Loom dialect pieces, exploration passes, ETG generation, materialization, one-shot bufferization, and TT-oriented cleanup passes.

### `loom-mlar`

Rust MLAR library for architecture description and schedule evaluation. The root install builds the 2D-mesh evaluator binary used by the pipeline unless MLAR is skipped or a compatible prebuilt evaluator is supplied.

### `loom2ttkernel`

Optional backend project for lowering bufferized Loom MLIR into TTKernel/tt-mlir flows. It is included as a submodule but is not built by `install-dev.sh` because it requires external Tenstorrent dependencies.

## Docker Development Environment

The recommended environment for developing the Tenstorrent backend is the
prebuilt [`ftod/loom_dev:latest`](https://hub.docker.com/r/ftod/loom_dev)
image on Docker Hub. The image is built from the repository
[Dockerfile](Dockerfile) and contains:

- the LLVM/MLIR 22 toolchain from the Tenstorrent tt-mlir CI image;
- tt-mlir at commit
  `5009f4764a31ff08e7cd838ed5747ae8a368a7e6`;
- TT-Metal source at commit
  `ad07818dd4d704d654359c8392794ef8ea8ceec4`;
- the tt-mlir compiler binaries and static libraries required by
  `loom2ttkernel`;
- Rust installed through rustup, plus the `tt-smi` and `tt-flash` tools.

The important paths are already exported in the image:

```text
MLIR_DIR=/opt/ttmlir-toolchain/lib/cmake/mlir
LLVM_DIR=/opt/ttmlir-toolchain/lib/cmake/llvm
TTMLIR_SOURCE_DIR=/opt/tt-mlir
TTMLIR_BUILD_DIR=/opt/tt-mlir/build
TT_METAL_HOME=/opt/tt-mlir/third_party/tt-metal/src/tt-metal
```

### 1. Install Docker and Get the Source

Docker Engine is required. The Buildx plugin is only needed when rebuilding
the image locally:

```bash
docker version
docker buildx version  # optional for Docker Hub users
```

Clone Loom together with all submodules:

```bash
git clone --recurse-submodules https://github.com/ecolab-nus/loom.git
cd loom
```

For an existing checkout, initialize or repair the submodules with:

```bash
git submodule update --init --recursive
```

### 2. Download or Build the Image

The normal path is to download the prebuilt image from Docker Hub:

```bash
docker pull ftod/loom_dev:latest
docker image inspect ftod/loom_dev:latest >/dev/null
```

The published image currently targets `linux/amd64`. After pulling it, proceed
directly to the verification and workspace build steps below; it is not
necessary to run `docker build`.

To rebuild the image locally after changing the Dockerfile, pull the
Tenstorrent base and assign the result a separate local tag:

```bash
docker pull ghcr.io/tenstorrent/tt-mlir/tt-mlir-ci-ubuntu-24-04:latest

docker buildx build \
  --load \
  --tag loom_dev:local \
  .
```

The first build downloads and compiles tt-mlir and can take several minutes.
Subsequent builds reuse the BuildKit cache. Expect roughly 16 GB of local disk
usage for the expanded image, in addition to BuildKit's compilation cache.

The pinned source revisions can be overridden deliberately:

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

To create such an archive on a machine that already has the image:

```bash
docker save ftod/loom_dev:latest | gzip >loom_dev-latest.tar.gz
```

Use `loom_dev:local` instead of `ftod/loom_dev:latest` in the commands below
when testing a locally rebuilt image.

### 3. Verify the Image

Run these checks before building Loom:

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

The two printed Git revisions should match the commits listed at the start of
this section.

### 4. Start the Development Container

From the Loom repository root, create a named container and mount the checkout
at `/workspace`:

```bash
docker run -it \
  --name loom-dev \
  --mount type=bind,src="$(pwd)",dst=/workspace \
  --workdir /workspace \
  ftod/loom_dev:latest
```

The named container keeps tools installed into its writable layer. Leave it
with `exit`, then resume the same environment later:

```bash
docker start --attach --interactive loom-dev
```

Use `--rm` instead of `--name loom-dev` when a disposable container is
preferred:

```bash
docker run --rm -it \
  --mount type=bind,src="$(pwd)",dst=/workspace \
  --workdir /workspace \
  ftod/loom_dev:latest
```

Commands in the container run as root, so newly generated files in the mounted
checkout may be owned by root on the host. If necessary, restore ownership
after exiting:

```bash
sudo chown -R "$(id -u):$(id -g)" .venv build third_party/*/build
```

Omit paths that do not exist.

### 5. Build the Complete Loom Workspace

The following commands run **inside the container**. Install `uv` once in a
named container:

```bash
python -m pip install "uv>=0.11"
uv --version
```

To discard native build products from a previous host or toolchain:

```bash
rm -rf \
  build \
  third_party/loom-dataflow/build \
  third_party/loom2ttkernel/build \
  third_party/loom-mlar/target
```

Prepare the locked Python 3.10 environment, build and install the
`loom-dataflow` Python extension, and build the MLAR evaluator:

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

Create a persistent CMake build of `loom-dataflow`. This supplies the static
libraries and generated dialect headers needed by `loom2ttkernel`:

```bash
bash third_party/loom-dataflow/build.sh \
  --mlir-dir="$MLIR_DIR"
```

Finally build the optional TTKernel backend against the tt-mlir artifacts in
the image:

```bash
bash third_party/loom2ttkernel/build.sh \
  -DMLIR_DIR="$MLIR_DIR" \
  -DLLVM_DIR="$LLVM_DIR" \
  -DTTMLIR_SOURCE_DIR="$TTMLIR_SOURCE_DIR" \
  -DTTMLIR_BUILD_DIR="$TTMLIR_BUILD_DIR"
```

Do not use `install-dev.sh` for this Docker workflow. That script is the native
host installation path and intentionally builds the pinned
`third_party/llvm-project` toolchain. The commands above reuse the LLVM/MLIR
installation already present in the image.

### 6. Smoke Tests and Running Loom

Verify the Python packages and generated native executables:

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

### 7. Run with Tenstorrent Hardware

LLVM/MLIR compilation does not require a Tenstorrent device. For hardware
access, the kernel driver and HugePages must be configured on the **host**;
the Dockerfile deliberately uses tt-installer's container mode and does not
modify the host.

First verify the host:

```bash
tt-smi
test -e /dev/tenstorrent
test -d /dev/hugepages-1G
```

Then pass all Tenstorrent devices and the HugePages mount into the container:

```bash
docker run -it \
  --name loom-dev-hw \
  --device /dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount type=bind,src="$(pwd)",dst=/workspace \
  --workdir /workspace \
  ftod/loom_dev:latest
```

Do not pass only an individual entry such as `/dev/tenstorrent/0`; Tenstorrent
documents that all devices must be passed through together. See the
[Tenstorrent Docker setup guide](https://docs.tenstorrent.com/tt-forge-onnx/getting_started_docker.html)
and the
[tt-installer container-mode guidance](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/tt-installer/)
for host setup details.

### 8. Common Docker Operations

```bash
# Show the image
docker image ls ftod/loom_dev:latest

# Show stopped and running development containers
docker ps --all --filter name=loom-dev

# Resume the named container
docker start --attach --interactive loom-dev

# Delete the named container; the bind-mounted source remains on the host
docker rm loom-dev

# Rebuild after changing the Dockerfile
docker buildx build --load --tag loom_dev:local .
```

## Native Quick Start

### Prerequisites

- [uv 0.11+](https://docs.astral.sh/uv/getting-started/installation/)

Optional, depending on which components you build:
- CMake >= 3.20, Ninja, lld, and a C++17 compiler for bundled LLVM/MLIR and `loom-dataflow`
- Rust toolchain for `loom-mlar`
- tt-metal and tt-mlir for `loom2ttkernel`

### Installation

Run the developer install script:

```bash
bash install-dev.sh
```

The script uses uv to install the pinned Python 3.10 interpreter when needed,
create `.venv`, and synchronize all Python packages from `uv.lock`. Do not
create or activate a separate Conda or virtualenv environment.

By default, the script initializes the `third_party/llvm-project` submodule at
the pinned commit `4efe170d858eb54432f520abb4e7f0086236748b`
(`LLVM 22.0.0git`) and incrementally builds LLVM/MLIR in
`build/llvm-4efe170d`. The generated MLIR CMake package is then used to build
`loom-dataflow`. The machine does not need a system LLVM or MLIR installation.
The LLVM source, build directory, and generated `MLIR_DIR` are derived from
paths inside the repository; no LLVM or MLIR path needs to be supplied.

### Install Script Options

```
bash install-dev.sh [OPTIONS]

Options:
  --skip-mlar         Skip building the loom-mlar Rust evaluator
  --skip-dataflow     Skip building loom-dataflow (C++ MLIR passes)
  --skip-helion       Skip installing helion-mlir
  --rebuild-dataflow  Force rebuilding loom-dataflow
  --help              Show help message

Environment variables:
  LOOM_LLVM_JOBS      Parallel LLVM build jobs (default: min(nproc, 32))
  LOOM_EVAL_SYSTEM    Path to a pre-built eval_system binary
```

The install has two stages:

1. Initialize the reusable Python 3.10 environment from `uv.lock`, without
   building `loom-dataflow`.
2. Incrementally build the pinned LLVM/MLIR toolchain, install
   `loom-dataflow`, and build the MLAR evaluator.

If a native build fails, stage 1 remains installed and is reused on the next
run. LLVM uses its existing Ninja build directory, and uv does not force a
`loom-dataflow` reinstall on every run. To deliberately rebuild the extension
after changing its C++ sources, run:

```bash
bash install-dev.sh --rebuild-dataflow
```

### Python Environment

Python dependencies are declared in the root and workspace-member
`pyproject.toml` files. Exact resolved versions are committed in `uv.lock`, and
`.python-version` pins the environment to Python 3.10.

To run only the first, Python-only stage without the native-system checks or
builds:

```bash
uv sync --locked --inexact --extra dataflow --extra helion \
  --no-install-package loom-dataflow
```

Run Python commands through uv so they always use the project environment:

```bash
uv run python --version
uv run pytest
```

Use `uv add`, `uv remove`, and `uv lock` when changing dependencies. Commit
`pyproject.toml` and `uv.lock` together.

## Usage

### Running a Kernel

Kernel scripts inherit a CLI from `LoomKernel`. The recommended path is a config file:

```bash
uv run python kernels/matmul.py --config kernels/config_files/matmul.json --njobs 16 --debug
```

Or pass paths explicitly:

```bash
uv run python kernels/matmul.py \
  --output-path test/matmul_2Dmesh \
  --hw-spec third_party/loom-mlar/tests/2d_mesh/2d_mesh_torus.mlir \
  --njobs 16 \
  --debug
```

### Configuration File

The config file is a JSON object specifying output and hardware paths, plus optional solver controls:

```json
{
  "output_path": "test/matmul_2Dmesh",
  "hw_spec": "third_party/loom-mlar/tests/2d_mesh/2d_mesh_torus.mlir",
  "block_sizes": {
    "tile_m": {"lb": 32, "ub": 256},
    "tile_n": {"lb": 32, "ub": 256}
  }
}
```

Use `assigned_block_size` to bypass the solver and materialize explicit assignments:

```json
{
  "output_path": "test/matmul_2Dmesh",
  "hw_spec": "third_party/loom-mlar/tests/2d_mesh/2d_mesh_torus.mlir",
  "assigned_block_size": {
    "ALL": {"tile_m": 128, "tile_n": 128, "tile_k": 64}
  }
}
```

### Writing a New Kernel

1. Create a Python file under `kernels/`.
2. Define a Helion kernel function and wrap it with `helion.kernel()`.
3. Subclass `LoomKernel`, set the `kernel` attribute, and implement `bind_args()`.
4. Add the standard `__main__` block.

```python
import torch
import helion
from loom import LoomKernel

def _my_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # ... helion kernel body ...
    pass

class MyKernel(LoomKernel):
    kernel_name = "my-kernel"
    kernel = helion.kernel(static_shapes=False)(_my_kernel)

    @classmethod
    def bind_args(cls):
        return (
            torch.randn([1024, 512], dtype=torch.float16),
            torch.randn([512, 1024], dtype=torch.float16),
        )

if __name__ == "__main__":
    MyKernel.run()
```

### Pipeline Output

After a successful run, the output directory contains:

```
<output_path>/
├── IRs/
│   ├── p00_from_helion_frontend.mlir   (--debug only)
│   ├── p01_explored.mlir               (--debug only)
│   └── p03_bufferized.mlir             (final output)
└── constraints/
    ├── p01_exploration_etg.json
    ├── p02_resolved_etg.json
    └── solver.log                      (--debug only)
```

## Scripts

| Script | Description |
|--------|-------------|
| `install-dev.sh` | Runs the two-stage Python and native developer installation. |
| `scripts/preflight.sh` | Checks uv, the pinned LLVM submodule, and required native build and Rust tools. |
| `scripts/build-llvm.sh` | Incrementally builds the pinned LLVM/MLIR commit for `loom-dataflow`. |
| `scripts/build-mlar.sh` | Builds the `loom-mlar` `eval_system` evaluator binary used by the root pipeline. |
