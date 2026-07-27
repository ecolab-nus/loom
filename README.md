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

Clone Loom with your preferred Git transport and initialize its submodules:

```bash
git submodule update --init --recursive
```

Install
[VS Code Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers),
open the cloned repository in VS Code, and run **Dev Containers: Reopen in
Container** from the Command Palette.

If Docker runs remotely, first connect to that host with
[Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh),
open the repository directory on that host, and run **Reopen in Container**
from the remote window.

VS Code pulls the development image, creates the container, mounts the
checkout, and runs `install-docker.sh` automatically.

### Without VS Code

On the Docker host:

```bash
docker pull ftod/loom_dev:latest
docker volume create loom-workspace

docker run -d \
  --name loom-dev \
  --hostname loom-dev \
  --init \
  --restart unless-stopped \
  --mount type=volume,src=loom-workspace,dst=/workspace \
  ftod/loom_dev:latest
```

Enter a local container with:

```bash
docker exec -it loom-dev bash
```

For Docker on a remote host:

```bash
ssh -t docker-host 'docker exec -it loom-dev bash'
```

Inside the container, install the workspace:

```bash
cd /workspace
git clone --recurse-submodules https://github.com/ecolab-nus/loom.git
cd loom
bash install-docker.sh
```

Then continue in the container shell. The standard VS Code workflow opens its
volume-backed workspace automatically, normally at `/workspaces/loom`. See
the [Docker development guide](docs/docker.md) for the Dev Container CLI,
local and remote access, lifecycle, build options, and hardware access.

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
- [Usage](docs/usage.md)
