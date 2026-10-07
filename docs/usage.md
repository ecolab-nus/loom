# Usage

Complete the [Docker development setup](docker.md) before running Loom.

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

Every tile size must divide its loop exactly; a grid tile's block count must
also divide the number of cores it is spread over. There is no tail handling,
so a shape with no exact tiling is infeasible. `block_sizes` bounds a tile's
candidates to the integers in `[lb, ub]`; under `LOOM_TARGET=tt`, tiles that
size the bottom two dimensions of an allocation must also be multiples of 32.

Use `assigned_block_size` to bypass the solver and materialize explicit
assignments. Loom still resolves the ETG and rejects each assignment that
violates a symbol domain, an exact loop extent, a hard constraint, or memory
capacity, or for which a primitive matches zero or several perf-model
alternatives. Valid variants continue; the run fails if none remain.

```json
{
  "output_path": "test/matmul_2Dmesh",
  "hw_spec": "third_party/loom-mlar/tests/2d_mesh/2d_mesh_torus.mlir",
  "assigned_block_size": {
    "ALL": {"tile_m": 128, "tile_n": 128, "tile_k": 64}
  }
}
```

## Automatic Processor Bindings

Enable automatic processor/memory binding directly on a Helion kernel with:

```bash
uv run python kernels/matmul.py \
  --config kernels/config_files/matmul.json \
  --enumerate-bindings
```

The equivalent config field is `"enumerate_bindings": true`.
`"explicit_memory": true` only means the input is already a stage-02 template;
it is not required for enumeration.

Binding also runs when enumeration is off: Loom chooses the last matching
registered processor implementation and fails if that fixed assignment or its
direct movers are infeasible; it does not fall back to an earlier registration.
Fixed mode preserves the ordinary spatial/broadcast candidate names. With
enumeration on, Loom explores legal combinations and adds deterministic binding
and mover suffixes. Each primitive in a fused `linalg.generic`
body is a binding site; the generic remains fused in IR. Unpinned residency
is chosen by binding. To pin an operand, name a platform memory:

- config: `"residency": {"k_view": "L1_S"}` pins every load of host tensor `k_view`;
- kernel: `set_memory_space(k_view[...], memory="L1_S")` pins one load.

Pins filter candidates in both binding modes. Unknown memories, `mem_`-prefixed
names, and numeric `local_mem_kind` encodings are rejected before lowering; an
unsatisfiable pin reports the operand and the memories candidates offer.
Authored stage-02 allocations name their memory with `on @<memory>`; omit
`loom.inferred_residency` to make that choice binding.

Kernel arguments live in the platform's DRAM-domain memory: the one memory whose
ADL op carries `domain = "DRAM"`, as emitted by the MLAR exporter. Platforms
exported before memory domains existed must be re-exported.

Loom rewrites internal allocations and supported aliases, then selects only
declared direct movers for transfers already present in the computation. A
binding that would need an implicit transfer is rejected even when a mover
exists. It does not synthesize routes or split shared allocations.

The solver writes `constraints/solver_results.json` containing every binding
group, every variant status and cost, the best feasible result per group, and
the overall best. Materialized IR preserves the selected compute and mover
identities.

Set `LOOM_TARGET=tt` to enable TT-specific storage accounting and
materialization rewrites. An unset or empty variable selects the generic
target; other values are rejected. The selected target is recorded as
`loom.target` in MLIR and in each ETG variant. Generic accounting uses dense
allocation bytes. TT accounting additionally enforces the bottom-two-dimension
rules, pads a static size-one storage dimension to 32 elements, includes
reduction configuration storage, and aligns tile sizes in the bottom two
allocation dimensions to 32.

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
