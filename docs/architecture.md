# Architecture

Loom is an end-to-end compilation stack for ML kernels targeting spatial
hardware architectures. Four subprojects implement the major compilation
layers, while the root Python package connects them into one workflow.

## Subprojects

| Subproject | Responsibility |
|------------|----------------|
| `third_party/helion-mlir` | Lowers Helion kernels into high-level MLIR with affine control flow and linalg-on-tensors compute. |
| `third_party/loom-dataflow` | Runs MLIR exploration and materialization passes, emits ETG JSON, and exposes the C++ pipeline through Python bindings. |
| `third_party/loom-mlar` | Describes hardware with MLAR and resolves ETG schedules against architecture performance models. |
| `third_party/loom2ttkernel` | Optionally lowers Loom-produced bufferized MLIR into TTKernel/tt-mlir/tt-metal code-generation inputs. |

## Root Repository

The root repository is the orchestration and active solver layer:

```text
loom-monorepo/
├── loom/                    # Python orchestration package
│   ├── pipeline.py          # End-to-end pipeline stages and output layout
│   ├── kernel_base.py       # LoomKernel base class and inherited kernel CLI
│   ├── solver/              # CPMpy/CP-SAT block-size optimizer
│   └── loom_utils/          # ETG, MLAR, modeling, and timing helpers
├── kernels/                 # Example kernel entrypoints and configurations
├── scripts/                 # Installation, preflight, and build helpers
├── install-docker.sh        # Docker workspace installation
├── test/                    # Generated and integration artifacts
├── tests/                   # Python regression tests
└── third_party/             # Loom subprojects and native dependencies
```

The root `loom` package asks `helion-mlir` for stage-00 MLIR, calls the
`loom-dataflow` Python bindings for exploration and materialization, sends ETG
variants through `loom-mlar`, and uses `loom.solver` to choose block-size
assignments. The solver consumes resolved ETG constraints and timing
expressions, searches finite symbol domains, and returns assignments for
materialization.

## Compilation Pipeline

| Stage | Name | Component | Description |
|-------|------|-----------|-------------|
| 0 | Helion Frontend | `helion-mlir` | Converts a bound Helion kernel into high-level MLIR. |
| 1 | Dataflow Exploration | `loom-dataflow` | Explores hardware mappings, annotates reuse and copy choices, and emits explored MLIR plus ETG JSON. |
| 2 | ETG Resolution | `loom-mlar` | Resolves ETG variants against architecture performance models. |
| 3 | Block-Size Solve | `loom.solver` | Uses CPMpy/CP-SAT to find feasible, low-cost block-size assignments. |
| 4 | Materialization | `loom-dataflow` | Applies selected block sizes and lowers tensor IR to bufferized Loom MLIR. |
| 5 | Optional TT Lowering | `loom2ttkernel` | Lowers bufferized Loom MLIR toward TTKernel code generation. |

When `assigned_block_size` is provided, the root pipeline bypasses Stage 3 and
sends those values directly to materialization. In debug mode, Loom can still
generate and resolve ETG data to produce manual latency breakdowns.

## Subproject Notes

### `helion-mlir`

This Python frontend starts from Helion Device IR FX graphs, maps control flow
to `affine.for` and `affine.parallel`, represents memory updates with tensor
IR, and uses torch-mlir for ATen/linalg lowering.

### `loom-dataflow`

This C++/MLIR project owns the ADL and Loom dialects, exploration passes, ETG
generation, materialization, one-shot bufferization, TT-oriented cleanup
passes, and pybind11 bindings.

### `loom-mlar`

This Rust MLAR library provides architecture descriptions and schedule
evaluation. The core installation builds the 2D-mesh evaluator used by the
pipeline unless MLAR is skipped or a compatible prebuilt evaluator is
supplied.

### `loom2ttkernel`

This optional backend lowers bufferized Loom MLIR into TTKernel/tt-mlir flows.
Its external Tenstorrent dependencies are supplied by the Docker development
environment, and it is built by `install-docker.sh`.
