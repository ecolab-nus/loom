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
└── third_party/             # The four subrepos listed above
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

- Python 3.10+

Optional, depending on which components you build:
- CMake >= 3.20, Ninja, lld, a C++17 compiler, and an MLIR installation for `loom-dataflow`
- Rust toolchain for `loom-mlar`
- tt-metal and tt-mlir for `loom2ttkernel`

### Installation

Create a Python 3.10 environment and run the developer install script:

```bash
conda create -n loom python=3.10 -y
conda activate loom

bash install-dev.sh
```

If you have a custom MLIR installation, pass it explicitly:

```bash
bash install-dev.sh --mlir-dir=/path/to/your/mlir/lib/cmake/mlir
```

or set `MLIR_DIR`:

```bash
export MLIR_DIR=/path/to/your/mlir/lib/cmake/mlir
bash install-dev.sh
```

### Install Script Options

```
bash install-dev.sh [OPTIONS]

Options:
  --mlir-dir=PATH     Path to MLIR cmake config directory
                       (default: $MLIR_DIR or /opt/llvm-mlir/lib/cmake/mlir)
  --skip-mlar         Skip building the loom-mlar Rust evaluator
  --skip-dataflow     Skip building loom-dataflow (C++ MLIR passes)
  --skip-helion       Skip installing helion-mlir
  --help              Show help message

Environment variables:
  PYTHON              Python 3.10+ interpreter used for every pip/install check
  MLIR_DIR            Path to MLIR cmake config directory
  LOOM_EVAL_SYSTEM    Path to a pre-built eval_system binary
```

The script initializes submodules, checks dependencies, installs `loom-dataflow` and `helion-mlir` in editable mode, builds the MLAR evaluator when available, and installs the root `loom` package.

## Usage

### Running a Kernel

Kernel scripts inherit a CLI from `LoomKernel`. The recommended path is a config file:

```bash
python kernels/matmul.py --config kernels/config_files/matmul.json --njobs 16 --debug
```

Or pass paths explicitly:

```bash
python kernels/matmul.py \
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
| `install-dev.sh` | Initializes submodules, runs preflight checks, installs core Python/C++ subprojects, and builds the MLAR evaluator when possible. |
| `scripts/preflight.sh` | Checks required Python, build, MLIR, and Rust dependencies. |
| `scripts/build-mlar.sh` | Builds the `loom-mlar` `eval_system` evaluator binary used by the root pipeline. |
