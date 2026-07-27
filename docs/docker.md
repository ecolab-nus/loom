# Docker Development

Loom follows the
[Development Container Specification](https://containers.dev/) through
`.devcontainer/devcontainer.json`. The configuration uses the prebuilt
[`ftod/loom_dev:latest`](https://hub.docker.com/r/ftod/loom_dev) image and
runs `install-docker.sh` when the development container is first created.

The recommended VS Code workflow starts from a checkout created with the
developer's preferred Git transport. VS Code mounts that checkout into the
development container.

## 1. Prerequisites

This guide uses two terms:

- **workstation**: the developer's local machine;
- **Docker host**: the machine running Docker Engine and the Loom container.

They may be the same machine.

The Docker host requires Docker Engine:

```bash
docker version
docker ps
```

For remote development, verify SSH access from the workstation:

```bash
ssh docker-host
```

Membership in the host's `docker` group typically grants root-equivalent
control of that host. Only grant Docker access to trusted developers.

VS Code users need:

- [Visual Studio Code](https://code.visualstudio.com/);
- [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers);
- [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh)
  when the Docker host is remote.

## 2. Dev Container Configuration

The repository configuration is:

```json
{
  "$schema": "https://raw.githubusercontent.com/devcontainers/spec/main/schemas/devContainer.base.schema.json",
  "name": "Loom",
  "image": "ftod/loom_dev:latest",
  "remoteUser": "root",
  "init": true,
  "overrideCommand": false,
  "postCreateCommand": "bash install-docker.sh"
}
```

It has the following behavior:

- pulls the published Loom development image;
- runs the container as `root`, matching the image toolchain;
- enables a small init process;
- keeps the image's `sleep infinity` command;
- creates Loom's `.venv` and builds all workspace components after cloning.

The image supplies `uv`, LLVM/MLIR, tt-mlir, TT-Metal, and Rust.
Project-specific Python packages are installed by the lifecycle command, not
baked into the image.

## 3. VS Code: Local Docker

Clone the repository by any preferred method, including SSH. Ensure its
submodules are initialized:

```bash
git submodule update --init --recursive
```

Then:

1. Open the repository directory in VS Code.
2. Open the Command Palette.
3. Run **Dev Containers: Reopen in Container**.
4. Wait for `postCreateCommand` to finish.

VS Code pulls the development image if necessary, creates the container,
bind-mounts the checkout, installs Loom, and opens the workspace inside the
container. Later, opening the same directory lets VS Code reopen the existing
development container.

## 4. VS Code: Remote Docker Host

Configure the Docker host in the workstation's `~/.ssh/config`:

```ssh-config
Host docker-host
  HostName YOUR_DOCKER_HOST
  User YOUR_REMOTE_USER
  IdentityFile ~/.ssh/id_ed25519
```

Then:

1. Run **Remote-SSH: Connect to Host...** and select `docker-host`.
2. Ensure Dev Containers is installed in the remote SSH window.
3. Clone Loom on the Docker host using the Git transport and credentials
   available there, unless it is already cloned.
4. Open that repository directory with **File: Open Folder...**.
5. Run **Dev Containers: Reopen in Container**.
6. Wait for the automatic workspace installation.

The Docker client does not need to be installed on the workstation. The
repository, container, and all build artifacts live on the remote Docker
host. Git credentials are used only on that host; the workflow does not
require an HTTPS repository URL.

## 5. Dev Container CLI

The reference `devcontainer` CLI can create the same declared environment
without VS Code. Install the CLI, then run it from an existing Loom checkout:

```bash
npm install --global @devcontainers/cli

cd /path/to/loom
git submodule update --init --recursive
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . bash
```

The CLI executes `postCreateCommand` during `up`. Its normal
`--workspace-folder` workflow bind-mounts the checkout from the Docker host.

## 6. Raw Docker CLI

Non-VS Code users can create the persistent environment directly on the
Docker host:

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

For a local Docker host:

```bash
docker exec -it loom-dev bash
```

For a remote Docker host, either log in first or combine both operations:

```bash
ssh docker-host
docker exec -it loom-dev bash
```

```bash
ssh -t docker-host 'docker exec -it loom-dev bash'
```

Inside a newly created raw Docker container:

```bash
cd /workspace
git clone --recurse-submodules https://github.com/ecolab-nus/loom.git
cd loom
bash install-docker.sh
```

Terminal editors, shells, and tools such as tmux can all run in this container
shell. No VS Code installation is required.

For a non-interactive remote command:

```bash
ssh docker-host \
  'docker exec loom-dev bash -lc "cd /workspace/loom && uv run python -c '\''import loom'\''"'
```

## 7. Installation Options

The automatic lifecycle command and raw Docker workflow both invoke:

```bash
bash install-docker.sh
```

Incremental reruns reuse existing Python, Cargo, and CMake artifacts.
Available options for manual runs are:

```text
bash install-docker.sh [OPTIONS]

Options:
  --skip-mlar         Skip the loom-mlar Rust evaluator
  --skip-dataflow     Skip loom-dataflow and loom2ttkernel
  --skip-helion       Skip helion-mlir
  --skip-ttkernel     Skip the optional loom2ttkernel backend
  --rebuild-dataflow  Force reinstalling the loom-dataflow Python extension
  --help              Show help

Environment:
  LOOM_EVAL_SYSTEM    Path to a prebuilt eval_system binary
```

Build products made with another toolchain should be removed before
installation:

```bash
rm -rf \
  build \
  third_party/loom-dataflow/build \
  third_party/loom2ttkernel/build \
  third_party/loom-mlar/target
```

## 8. Smoke Tests

Run inside the Loom workspace:

```bash
uv run python -c 'import loom, loom_pipeline, helion_mlir'

test -x third_party/loom-mlar/tests/2d_mesh/bin/eval_system
test -x third_party/loom2ttkernel/build/bin/tileloom_to_ttkernel_opt

uv run pytest
```

Run the example used to validate the published environment:

```bash
uv run python kernels/matmul.py \
  --config kernels/config_files/matmul.json \
  --njobs 16 \
  --debug
```

See the [usage guide](usage.md) for configuration and output details.

## 9. Raw Container Lifecycle

Dev Container-aware tools manage their own containers. For the named
`loom-dev` container created by the raw Docker workflow:

```bash
docker stop loom-dev
docker start loom-dev
docker exec -it loom-dev bash
```

Deleting the container does not delete its named volume:

```bash
docker rm --force loom-dev
```

Recreate it with the command in section 6 and the same `loom-workspace`
volume.

Do not remove the volume unless you intend to delete the checkout, Python
environment, caches, and all build output:

```bash
docker volume rm loom-workspace
```

## 10. Tenstorrent Hardware Access

LLVM/MLIR compilation does not require a Tenstorrent device. For hardware
access, configure the kernel driver and HugePages on the Docker host.

The raw Docker command with hardware access is:

```bash
docker run -d \
  --name loom-dev \
  --hostname loom-dev \
  --init \
  --restart unless-stopped \
  --device /dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount type=volume,src=loom-workspace,dst=/workspace \
  ftod/loom_dev:latest
```

Pass the complete `/dev/tenstorrent` device set rather than one numbered
entry.

For Dev Container tools, add the equivalent device and HugePages arguments to
a local override of `devcontainer.json` when hardware access is needed.

## 11. Image Details and Local Rebuilds

The image's toolchain paths and pinned source revisions are defined in the
repository
[Dockerfile](https://github.com/ecolab-nus/loom/blob/main/Dockerfile).
`MLIR_DIR`, `LLVM_DIR`, and related variables are available in Dev Container
and `docker exec` sessions.

Buildx is required only when rebuilding the image:

```bash
docker buildx build \
  --load \
  --tag loom_dev:local \
  .
```

Use `loom_dev:local` in a temporary local copy of `devcontainer.json` while
testing a locally rebuilt image.

For an offline machine, create and transfer an archive:

```bash
docker save ftod/loom_dev:latest | gzip >loom_dev-latest.tar.gz
gzip -dc loom_dev-latest.tar.gz | docker load
```
