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

Install Docker, an OpenSSH client, and
[VS Code Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh).
Create an SSH key if you do not already have one:

```bash
test -f ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519
```

Pull the development image and start a persistent SSH container. Loom's source
and build environment live in the `loom-workspace` Docker volume:

```bash
docker pull ftod/loom_dev:latest
docker volume create loom-workspace

docker run -d \
  --name loom-dev \
  --hostname loom-dev \
  --restart unless-stopped \
  --publish 127.0.0.1:2222:22 \
  --mount type=volume,src=loom-workspace,dst=/workspace \
  --mount type=bind,src="$HOME/.ssh/id_ed25519.pub",dst=/run/loom/authorized_key,readonly \
  ftod/loom_dev:latest
```

Add the following host to `~/.ssh/config`:

```ssh-config
Host loom-dev
  HostName 127.0.0.1
  Port 2222
  User root
  IdentityFile ~/.ssh/id_ed25519
```

In VS Code, run **Remote-SSH: Connect to Host...**, choose `loom-dev`, and
open a terminal to install the workspace:

```bash
cd /workspace
git clone --recurse-submodules https://github.com/ecolab-nus/loom.git
cd loom
bash install-docker.sh
```

Then open `/workspace/loom` in VS Code. You can also connect with
`ssh loom-dev`. See the [Docker development guide](docs/docker.md) for
container lifecycle, build options, image details, and hardware access.

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
