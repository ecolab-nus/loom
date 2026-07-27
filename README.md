<img src="assets/loom-logo.svg" alt="Loom Logo" width="250">

# Loom

Loom is an end-to-end compilation stack for mapping ML kernels onto spatial
hardware architectures. It connects a Helion frontend, MLIR dataflow
exploration, MLAR architecture modeling, constraint solving, and optional
Tenstorrent TTKernel lowering in one workflow.

```text
Helion → MLIR exploration → MLAR resolution → block-size solve
       → materialization → optional TTKernel lowering
```

See the [architecture guide](docs/architecture.md) for the compilation stages
and repository layout.

## Quick Start

Clone the repository and its submodules:

```bash
git clone --recurse-submodules https://github.com/ecolab-nus/loom.git
cd loom
```

### Docker

Pull the development image and mount the repository:

```bash
docker pull ftod/loom_dev:latest

docker run -it \
  --name loom-dev \
  --mount type=bind,src="$(pwd)",dst=/workspace \
  --workdir /workspace \
  ftod/loom_dev:latest
```

Continue with the [Docker workspace build](docs/docker.md#5-build-the-workspace).

### Native

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then
run:

```bash
bash install-dev.sh
```

See the [native development guide](docs/development.md) for prerequisites and
installation options.

## Run an Example

```bash
uv run python kernels/matmul.py \
  --config kernels/config_files/matmul.json \
  --njobs 16 \
  --debug
```

See the [usage guide](docs/usage.md) for kernel configuration, writing new
kernels, and generated outputs.

## Documentation

Preview the Docusaurus site locally with Node.js 20 or newer:

```bash
cd website
npm install
npm start
```

Then open [http://localhost:3000](http://localhost:3000). Use
`npm run build` to create the production site under `website/build/`.

- [Documentation index](docs/README.md)
- [Architecture and compilation pipeline](docs/architecture.md)
- [Docker development](docs/docker.md)
- [Native development](docs/development.md)
- [Usage](docs/usage.md)
