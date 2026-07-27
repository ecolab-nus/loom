# Native Development

This guide describes installation directly on the host. For the recommended
Tenstorrent backend environment, see [Docker development](docker.md).

## Prerequisites

- [uv 0.11+](https://docs.astral.sh/uv/getting-started/installation/)

Optional requirements depend on the components being built:

- CMake 3.20 or newer, Ninja, lld, and a C++17 compiler for LLVM/MLIR and
  `loom-dataflow`;
- a Rust toolchain for `loom-mlar`;
- tt-metal and tt-mlir for `loom2ttkernel`.

## Installation

Clone the repository with its submodules and run the developer installer:

```bash
git clone --recurse-submodules https://github.com/ecolab-nus/loom.git
cd loom
bash install-dev.sh
```

For an existing checkout:

```bash
git submodule update --init --recursive
bash install-dev.sh
```

The script uses uv to install the pinned Python 3.10 interpreter when needed,
creates `.venv`, and synchronizes the Python packages from `uv.lock`. Do not
create or activate a separate Conda or virtualenv environment.

By default, the script initializes the repository's pinned LLVM/MLIR source
and incrementally builds it under `build/`. The generated MLIR CMake package
is then used to build `loom-dataflow`; no system LLVM or MLIR installation is
required.

## Installer Options

```text
bash install-dev.sh [OPTIONS]

Options:
  --skip-mlar         Skip building the loom-mlar Rust evaluator
  --skip-dataflow     Skip building loom-dataflow (C++ MLIR passes)
  --skip-helion       Skip installing helion-mlir
  --rebuild-dataflow  Force rebuilding loom-dataflow
  --help              Show help

Environment variables:
  LOOM_LLVM_JOBS      Parallel LLVM build jobs (default: min(nproc, 32))
  LOOM_EVAL_SYSTEM    Path to a prebuilt eval_system binary
```

The installation has two stages:

1. Initialize the reusable Python environment from `uv.lock` without building
   `loom-dataflow`.
2. Incrementally build LLVM/MLIR, install `loom-dataflow`, and build the MLAR
   evaluator.

If a native build fails, the first-stage environment remains installed and is
reused on the next run. To deliberately rebuild the dataflow extension after
changing its C++ sources:

```bash
bash install-dev.sh --rebuild-dataflow
```

## Python Environment

Python dependencies are declared in the root and workspace-member
`pyproject.toml` files. Exact resolved versions are committed in `uv.lock`,
and `.python-version` pins Python 3.10.

Run only the Python stage without native builds:

```bash
uv sync --locked --inexact --extra dataflow --extra helion \
  --no-install-package loom-dataflow
```

Run Python commands through uv:

```bash
uv run python --version
uv run pytest
```

Use `uv add`, `uv remove`, and `uv lock` when changing dependencies. Commit
`pyproject.toml` and `uv.lock` together.

## Build Scripts

| Script | Description |
|--------|-------------|
| `install-dev.sh` | Runs the two-stage Python and native developer installation. |
| `scripts/preflight.sh` | Checks uv and the required native and Rust tools. |
| `scripts/build-llvm.sh` | Incrementally builds the pinned LLVM/MLIR toolchain for `loom-dataflow`. |
| `scripts/build-mlar.sh` | Builds the `loom-mlar` `eval_system` evaluator used by the root pipeline. |

