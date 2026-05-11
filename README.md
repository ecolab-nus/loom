<img src="assets/loom-logo.svg" alt="Loom Logo" width="250">

# Loom Monorepo

Loom is an end-to-end compilation pipeline for ML kernels targeting spatial hardware architectures. It takes high-level kernel descriptions (written in [Helion](https://github.com/pytorch-labs/helion)) and compiles them through a multi-stage pipeline — MLIR lowering, dataflow exploration, symbolic architecture evaluation, and SMT-based block-size optimization — to produce optimized, bufferized MLIR ready for code generation.

## Repository Structure

```
loom-monorepo/
├── loom/                    # Root Python package — pipeline orchestrator & SMT solver
│   ├── pipeline.py          # End-to-end pipeline (Steps 0–4)
│   ├── kernel_base.py       # LoomKernel base class with built-in CLI
│   ├── smt/                 # Z3-based SMT solver for block-size optimization
│   └── loom_utils/          # MLAR evaluator bridge, ETG resolver, timers
├── kernels/                 # Example kernel scripts (e.g., matmul)
│   ├── matmul.py            # Matrix-multiply kernel using Helion + LoomKernel
│   └── config.json          # Sample configuration file
├── scripts/                 # Developer scripts
│   ├── preflight.sh         # Pre-flight dependency checker
│   └── build-mlar.sh        # Builds the loom-mlar eval_core binary
├── install-dev.sh           # One-click developer install
├── test/                    # Integration test artifacts
└── third_party/             # Git submodules
    ├── helion-mlir/         # Python: Helion kernel → MLIR frontend
    ├── loom-dataflow/       # C++/Python: MLIR exploration & materialization passes
    └── loom-mlar/           # Rust: architecture modeling & symbolic evaluator
```

## Compilation Pipeline

The Loom pipeline consists of five stages:

| Stage | Name | Component | Description |
|-------|------|-----------|-------------|
| 0 | **Helion Frontend** | `helion-mlir` | Converts a Helion kernel into high-level MLIR (affine + linalg-on-tensors) |
| 1 | **Exploration** | `loom-dataflow` | Applies C++ MLIR passes to explore hardware mappings and produce an Exploration Task Graph (ETG) |
| 2 | **ETG Resolution** | `loom-mlar` | Evaluates ETG variants against a symbolic architecture model via the Rust evaluator |
| 3 | **SMT Solver** | `loom.smt` | Uses Z3 to find optimal block sizes satisfying all hardware constraints |
| 4 | **Materialization** | `loom-dataflow` | Applies the solved block sizes and lowers MLIR to bufferized form |

## Third-Party Submodules

### helion-mlir

A Python frontend that lowers Helion kernels (Device IR FX graphs) into high-level MLIR with affine and linalg-on-tensors dialects. It maps Helion control flow to `affine.for`/`affine.parallel`, converts memory operations to tensor IR, and integrates torch-mlir for ATen operation lowering. This replaces Helion's default Triton lowering with a more architecture-friendly IR.

### loom-dataflow

The core MLIR-backed compiler infrastructure for exploring hardware scale-out models and dataflow patterns. It provides a custom MLIR `df` dialect for describing spatial dimensions and interconnect topologies, C++ passes that affinize kernels, tile affine loops, enumerate spatial hardware mappings, and analyze reuse patterns. Built as a C++ library with pybind11 bindings exposed to Python.

### loom-mlar

A Rust library implementing the Multi-Level Architecture Representation (MLAR) for composable, symbolic hardware description. It supports recursive architecture composition (Unit → Array → Graph), symbolic performance modeling with constraints, and generates an evaluator binary (`eval_core`) that accepts Schedule JSON on stdin and outputs evaluated performance scenarios.

## Quick Start

### Prerequisites

- Python 3.10+

Optional (for building all components from source):
- CMake ≥ 3.20, Ninja, lld, a C++17 compiler, and a pre-built MLIR installation (for `loom-dataflow`)
- Rust toolchain (for `loom-mlar`)

### Installation

Create a Python 3.10 environment and run the one-click install script:

```bash
# Using conda (recommended)
conda create -n loom python=3.10 -y
conda activate loom

# Install everything
bash install-dev.sh
```

That's it. The install script handles git submodule initialization, dependency checks, and building/installing all subprojects in the correct order.

If you have a custom MLIR installation, pass the path with `--mlir-dir`:

```bash
bash install-dev.sh --mlir-dir=/path/to/your/mlir/lib/cmake/mlir
```

Alternatively, set the `MLIR_DIR` environment variable:

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
  MLIR_DIR            Path to MLIR cmake config directory
  LOOM_EVAL_CORE      Path to a pre-built eval_core binary (skips the Rust build)
```

The script will automatically detect missing optional dependencies and skip the corresponding components with a warning, so you can get started even without CMake or Rust installed.

## Usage

### Running a Kernel

Kernel scripts inherit a full CLI from `LoomKernel`. The recommended way is to use a config file:

```bash
python kernels/matmul.py --config kernels/config.json --njobs 16 --debug
```

Or pass paths explicitly:

```bash
python kernels/matmul.py \
    --output-path test/mm_2Dmesh \
    --df-mlir third_party/loom-dataflow/test/Dialect/DataflowDialect/2D_mesh.mlir \
    --hw-compute-dir third_party/loom-mlar/tests/2d_mesh/compute \
    --njobs 16 --debug
```

### Configuration File

The config file is a JSON object specifying hardware paths and optional overrides:

```json
{
    "output_path": "test/mm_2Dmesh",
    "df_mlir": "third_party/loom-dataflow/test/Dialect/DataflowDialect/2D_mesh.mlir",
    "hw_compute_dir": "third_party/loom-mlar/tests/2d_mesh/compute",
    "block_sizes": {
        "block_size_0": 128,
        "block_size_1": 32,
        "block_size_2": 128
    }
}
```

When `block_sizes` is provided, Steps 2 and 3 (ETG resolution and SMT solving) are skipped, and the given values are used directly for materialization.

### Writing a New Kernel

1. Create a new Python file under `kernels/`.
2. Define a Helion kernel function and wrap it with `helion.kernel()`.
3. Subclass `LoomKernel`, set the `kernel` attribute, and implement `bind_args()`.
4. Add the standard `__main__` block.

```python
import torch
import helion
import helion.language as hl
from loom import LoomKernel

def _my_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # ... helion kernel body ...
    pass

class MyKernel(LoomKernel):
    kernel_name = "my-kernel"
    kernel = helion.kernel(static_shapes=False)(_my_kernel)

    @classmethod
    def bind_args(cls):
        return (torch.randn([1024, 512], dtype=torch.float16),
                torch.randn([512, 1024], dtype=torch.float16))

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
    └── smt_solver.log                  (--debug only)
```

## Scripts

| Script | Description |
|--------|-------------|
| `install-dev.sh` | One-click developer install — initializes submodules, runs pre-flight checks, and installs all subprojects in editable mode |
| `scripts/preflight.sh` | Checks for all required dependencies (Python, pip, cmake, ninja, lld, C++ compiler, MLIR, Rust) and reports what is missing |
| `scripts/build-mlar.sh` | Builds the `loom-mlar` eval_core evaluator binary via `cargo test` and copies it to `third_party/loom-mlar/bin/eval_core` |
