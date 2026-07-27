# Usage

Complete either the [Docker](docker.md) or
[native](development.md) development setup before running Loom.

## Running a Kernel

Kernel scripts inherit their command-line interface from `LoomKernel`. The
recommended path is a configuration file:

```bash
uv run python kernels/matmul.py \
  --config kernels/config_files/matmul.json \
  --njobs 16 \
  --debug
```

Paths can also be passed explicitly:

```bash
uv run python kernels/matmul.py \
  --output-path test/matmul_2Dmesh \
  --hw-spec third_party/loom-mlar/tests/2d_mesh/2d_mesh_torus.mlir \
  --njobs 16 \
  --debug
```

## Configuration Files

A configuration is a JSON object containing output and hardware paths plus
optional solver controls:

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

Use `assigned_block_size` to bypass the solver and materialize explicit
assignments:

```json
{
  "output_path": "test/matmul_2Dmesh",
  "hw_spec": "third_party/loom-mlar/tests/2d_mesh/2d_mesh_torus.mlir",
  "assigned_block_size": {
    "ALL": {"tile_m": 128, "tile_n": 128, "tile_k": 64}
  }
}
```

## Writing a Kernel

1. Create a Python file under `kernels/`.
2. Define a Helion kernel function and wrap it with `helion.kernel()`.
3. Subclass `LoomKernel`, set the `kernel` attribute, and implement
   `bind_args()`.
4. Add the standard `__main__` block.

```python
import helion
import torch

from loom import LoomKernel


def _my_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # ... Helion kernel body ...
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

## Pipeline Output

After a successful run, the output directory contains:

```text
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

