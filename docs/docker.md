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
  "updateRemoteUserUID": false,
  "init": true,
  "overrideCommand": false,
  "workspaceMount": "source=${localWorkspaceFolder},target=/workspace/${localWorkspaceFolderBasename},type=bind",
  "workspaceFolder": "/workspace/${localWorkspaceFolderBasename}",
  "postCreateCommand": "git config --global --add safe.directory '*' && bash install-docker.sh"
}
```

It has the following behavior:

- pulls the published Loom development image;
- runs VS Code terminals and workspace setup as `root`, avoiding
  container-internal permission issues;
- trusts bind-mounted Git repositories and submodules regardless of their
  host-side ownership;
- enables a small init process;
- keeps the image's `sleep infinity` command;
- creates Loom's `.venv` and builds all workspace components after cloning.

The standard image supplies `uv`, LLVM/MLIR, tt-mlir, TT-Metal sources, and Rust.
The Tenstorrent image additionally compiles the pinned TT-Metal checkout with
Python 3.10 bindings and its matching SFPI compiler. It stores a TT-NN wheel in
`/opt/loom/wheels`; `install-docker.sh` installs this wheel into the workspace's
`.venv`, replacing any separately installed PyPI TT-NN and reusing an already
installed matching wheel. The standard image does not build or install TT-NN.
Project-specific Python packages are installed by the lifecycle command, not
baked into the image.

The workspace uses CPU-only PyTorch with Helion, Triton, and torch-mlir for
CPU tensor tracing and MLIR lowering. It does not install the NVIDIA CUDA
runtime packages required by CUDA-enabled PyTorch. Running or autotuning
Helion kernels on NVIDIA GPUs requires a separate CUDA-enabled environment.

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

Non-VS Code users can work over a normal SSH connection. From a Loom checkout
on the Docker host, run:

```bash
./docker/start-container.sh
```

The launcher:

- pulls `ftod/loom_dev:latest`;
- creates the persistent `loom-workspace` Docker volume;
- creates `loom-dev` with container port 22 published only at
  `127.0.0.1:2222` on the Docker host;
- copies the current Docker host account's `~/.ssh/authorized_keys` into the
  container;
- starts the container.

Only public authorization data is copied. Private SSH keys are never copied.
The image disables password login and accepts only the copied public keys.

Connect from the Docker host:

```bash
ssh -A -p 2222 root@localhost
```

When Docker is remote, connect to the host and then to the container:

```bash
ssh -A docker-host
ssh -A -p 2222 root@localhost
```

The same connection can be made directly from the workstation:

```bash
ssh -A -J docker-host \
  -o HostKeyAlias=loom-dev-on-docker-host \
  -p 2222 root@localhost
```

The copied file represents the account that ran the launcher. If developers
use separate Unix accounts on the Docker host, each container inherits only
the keys authorized for its owning account.

Inside a newly created container, clone with any Git transport and install:

```bash
cd /workspace
git clone --recurse-submodules YOUR_GIT_URL loom
cd loom
bash install-docker.sh
```

The `-A` option forwards the workstation's SSH agent. If `YOUR_GIT_URL` uses
SSH, first load the corresponding key into the workstation's agent. The
forwarded agent lets Git authenticate without placing a private key on either
the Docker host or the container.

For a non-interactive remote command:

```bash
ssh -A -J docker-host \
  -o HostKeyAlias=loom-dev-on-docker-host \
  -p 2222 root@localhost \
  'cd /workspace/loom && uv run python -c "import loom"'
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
  --clean             Remove generated files before installing
  --clean-only        Remove generated files and exit
  --skip-mlar         Skip the loom-mlar Rust evaluator
  --skip-dataflow     Skip loom-dataflow and loom2ttkernel
  --skip-helion       Skip helion-mlir
  --skip-ttkernel     Skip the optional loom2ttkernel backend
  --rebuild-dataflow  Force reinstalling the loom-dataflow Python extension
  --help              Show help

Environment:
  LOOM_EVAL_SYSTEM    Path to a prebuilt eval_system binary
```

The installer builds the standalone ADL dialect first and supplies its CMake
package to both `loom-dataflow` and `loom2ttkernel`. Build products made with
another toolchain can be removed automatically:

```bash
bash install-docker.sh --clean
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
ssh -A -p 2222 root@localhost
```

Deleting the container does not delete its named volume:

```bash
docker rm --force loom-dev
```

Recreate it with the command in section 6 and the same `loom-workspace`
volume.

If the Docker host's authorized keys change, refresh the container copy and
restart SSH:

```bash
docker cp ~/.ssh/authorized_keys loom-dev:/run/loom/authorized_keys
docker restart loom-dev
```

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
[Dockerfile](https://github.com/ecolab-nus/loom/blob/main/docker/Dockerfile).
`MLIR_DIR`, `LLVM_DIR`, and related variables are available in Dev Container
and `docker exec` sessions.

Buildx is required only when rebuilding the image:

```bash
docker buildx build \
  --file docker/Dockerfile \
  --target standard \
  --load \
  --tag loom_dev:local \
  .
```

Use `loom_dev:local` in a temporary local copy of `devcontainer.json` while
testing a locally rebuilt image.

For Tenstorrent hardware, build the separate runtime image. This target extends
the published Loom image pinned by digest in `LOOM_BASE_IMAGE`, reuses its
prebuilt tt-mlir, and compiles its existing matching TT-Metal checkout. It does
not rebuild the standard image or tt-mlir. Before installing build dependencies,
it refreshes the Tenstorrent APT signing key from the official repository so
an older key in the pinned base does not cause `NO_PUBKEY` failures:

```bash
docker buildx build \
  --file docker/Dockerfile \
  --target tenstorrent \
  --load \
  --tag ftod/loom_dev:tenstorrent \
  .
```

The Tenstorrent Dev Container configuration builds this target automatically
when you select **Loom (Tenstorrent)** and reopen the checkout in a container.
On a new machine, it pulls the pinned prebuilt Loom base and builds only the
matching TT-Metal runtime; no manual image build or published
`ftod/loom_dev:tenstorrent` tag is required. The first build takes time;
subsequent builds reuse Docker's cache. The Docker host must already have the
Tenstorrent driver, `/dev/tenstorrent`, `/dev/hugepages`, and
`/dev/hugepages-1G` configured.

Raw Docker users can use the manual build command above and select
`ftod/loom_dev:tenstorrent` when creating their container.
An untargeted build defaults to the standard image. Limit build memory usage
with `--build-arg CMAKE_BUILD_PARALLEL_LEVEL=4` if needed. To select another
prebuilt Loom image, pass `--build-arg LOOM_BASE_IMAGE=IMAGE_REFERENCE`; that
image must include the compiled tt-mlir and its matching TT-Metal sources.

After recreating the container, run `bash install-docker.sh` (automatic for
Dev Containers), including when reusing a workspace volume. Do not install a
separate PyPI `ttnn` wheel. The runtime uses the checkout's verified SFPI under
`$TT_METAL_HOME/runtime/sfpi`, ahead of the system compiler supplied by the base
image. NumPy is constrained to 1.x in the workspace lockfile to satisfy this
TT-NN revision and prevent `uv run` from replacing it with NumPy 2.x.

The image build checks the SFPI executable and imports the built TT-NN wheel
without hardware. Device initialization and matmul still need validation on
the Docker host's Tenstorrent devices.

For an offline machine, create and transfer an archive:

```bash
docker save ftod/loom_dev:latest | gzip >loom_dev-latest.tar.gz
gzip -dc loom_dev-latest.tar.gz | docker load
```
