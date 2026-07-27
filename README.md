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

On the Docker host, run the provided launcher from a Loom checkout:

```bash
./docker/start-container.sh
```

It copies the Docker host account's `~/.ssh/authorized_keys` into the
container. Connect from the Docker host:

```bash
ssh -A -p 2222 root@localhost
```

If Docker runs remotely, either connect in two steps:

```bash
ssh -A docker-host
ssh -A -p 2222 root@localhost
```

or directly through the Docker host:

```bash
ssh -A -J docker-host \
  -o HostKeyAlias=loom-dev-on-docker-host \
  -p 2222 root@localhost
```

Inside the container, clone with your preferred Git URL and install:

```bash
cd /workspace
git clone --recurse-submodules YOUR_GIT_URL loom
cd loom
bash install-docker.sh
```

`-A` forwards the workstation's SSH agent, allowing SSH Git access without
copying a private key into the container. See the
[Docker development guide](docs/docker.md) for details and lifecycle commands.

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
