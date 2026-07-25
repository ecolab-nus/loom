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

## Quick Start

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
the pinned commit `6ad25c5912fcf13b44fcc03bd6a66dc33348cd68`
(`LLVM 22.0.0git`) and incrementally builds LLVM/MLIR in
`build/llvm-6ad25c59`. The generated MLIR CMake package is then used to build
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
