# Docker Development

The supported development environment is the prebuilt
[`ftod/loom_dev:latest`](https://hub.docker.com/r/ftod/loom_dev) image. Run it
as a persistent SSH server and keep the checkout in a Docker named volume.
This gives command-line tools and VS Code direct access to the same container
without bind-mounting the repository from the host.

## 1. Prerequisites

Install:

- Docker Engine;
- an OpenSSH client;
- [Visual Studio Code](https://code.visualstudio.com/);
- the
  [Remote - SSH extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh).

Verify Docker:

```bash
docker version
```

The container accepts SSH keys only. Create an Ed25519 key if the default key
does not exist:

```bash
test -f ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519
```

## 2. Download the Image

```bash
docker pull ftod/loom_dev:latest
docker image inspect ftod/loom_dev:latest >/dev/null
```

The published image currently targets `linux/amd64`.

To verify its main tools without starting the SSH server:

```bash
docker run --rm ftod/loom_dev:latest bash -lc '
  set -e
  test -f "$MLIR_DIR/MLIRConfig.cmake"
  test -f "$LLVM_DIR/LLVMConfig.cmake"
  test -f "$TTMLIR_BUILD_DIR/lib/libMLIRTTKernelDialect.a"
  ttmlir-opt --version
  mlir-opt --version
  rustc --version
  cargo --version
  uv --version
'
```

## 3. Start the Persistent Container

Create a named volume for the source tree, Python environment, caches, and
build products:

```bash
docker volume create loom-workspace
```

Start the container. Publishing SSH only on `127.0.0.1` prevents remote hosts
from connecting directly:

```bash
docker run -d \
  --name loom-dev \
  --hostname loom-dev \
  --restart unless-stopped \
  --publish 127.0.0.1:2222:22 \
  --mount type=volume,src=loom-workspace,dst=/workspace \
  --mount type=bind,src="$HOME/.ssh/id_ed25519.pub",dst=/run/loom/authorized_key,readonly \
  ftod/loom_dev:latest
```

Check that the SSH server is ready:

```bash
docker logs loom-dev
docker ps --filter name=loom-dev
```

If you use a different key, change both the public-key path in `docker run`
and the private-key path in the SSH configuration below.

## 4. Configure SSH

Add this entry to `~/.ssh/config` on the host:

```ssh-config
Host loom-dev
  HostName 127.0.0.1
  Port 2222
  User root
  IdentityFile ~/.ssh/id_ed25519
```

Test the connection:

```bash
ssh loom-dev
```

The image uses public-key authentication and disables SSH passwords. The
mounted public key is copied into the container with restrictive permissions
when it starts.

## 5. Develop with VS Code

VS Code Remote SSH is the recommended interface:

1. Open the Command Palette.
2. Run **Remote-SSH: Connect to Host...**.
3. Select `loom-dev`.
4. After connecting, choose **Open Folder...** and open `/workspace/loom`.

On the first connection, use the VS Code terminal to clone and install Loom:

```bash
cd /workspace
git clone --recurse-submodules https://github.com/ecolab-nus/loom.git
cd loom
bash install-docker.sh
```

VS Code installs its remote server and workspace extensions inside the
container. The source, `.venv`, uv-managed Python, uv cache, and compiled
artifacts remain in `loom-workspace`, so they survive container restarts and
container replacement as long as the same volume is mounted at `/workspace`.

## 6. Build the Workspace

Run the installer from `/workspace/loom` in an SSH or VS Code terminal:

```bash
bash install-docker.sh
```

The image supplies `uv`, while `install-docker.sh` creates Loom's project
Python environment under `.venv` and builds `loom-dataflow`, `loom-mlar`, and
`loom2ttkernel`.

Incremental reruns reuse the existing Python, Cargo, and CMake artifacts.
Available options are:

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

## 7. Smoke Tests

Run inside the container:

```bash
uv run python -c 'import loom, loom_pipeline, helion_mlir'

test -x third_party/loom-mlar/tests/2d_mesh/bin/eval_system
test -x third_party/loom2ttkernel/build/bin/tileloom_to_ttkernel_opt

uv run pytest
```

Run an example:

```bash
uv run python kernels/matmul.py \
  --config kernels/config_files/matmul.json \
  --njobs 16 \
  --debug
```

See the [usage guide](usage.md) for configuration and output details.

## 8. Container Lifecycle

Stop and resume the environment without losing work:

```bash
docker stop loom-dev
docker start loom-dev
ssh loom-dev
```

Inspect it:

```bash
docker logs loom-dev
docker ps --all --filter name=loom-dev
```

Deleting the container does not delete the named volume:

```bash
docker rm --force loom-dev
```

You can then recreate the container with the command in section 3 and the
same `loom-workspace` volume.

Do not remove `loom-workspace` unless you intend to delete the checkout,
Python environment, caches, and all build output stored in it:

```bash
docker volume rm loom-workspace
```

If recreating the container changes its SSH host key, clear the old local
entry and reconnect:

```bash
ssh-keygen -R '[127.0.0.1]:2222'
```

## 9. Tenstorrent Hardware Access

LLVM/MLIR compilation does not require a Tenstorrent device. For hardware
access, configure the kernel driver and HugePages on the host, then add the
device and HugePages mounts when creating the persistent container:

```bash
docker run -d \
  --name loom-dev \
  --hostname loom-dev \
  --restart unless-stopped \
  --publish 127.0.0.1:2222:22 \
  --device /dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount type=volume,src=loom-workspace,dst=/workspace \
  --mount type=bind,src="$HOME/.ssh/id_ed25519.pub",dst=/run/loom/authorized_key,readonly \
  ftod/loom_dev:latest
```

Pass the complete `/dev/tenstorrent` device set rather than one numbered
entry.

## 10. Image Details and Local Rebuilds

The image contains the pinned LLVM/MLIR, tt-mlir, TT-Metal, and Rust
toolchains required by Loom. Their paths and revisions are defined in the
repository
[Dockerfile](https://github.com/ecolab-nus/loom/blob/main/Dockerfile).
`MLIR_DIR`, `LLVM_DIR`, and related toolchain variables are available in both
Docker commands and SSH sessions. Loom's project-specific Python environment
is not preinstalled; only `uv` is provided for creating it.

Buildx is required only when rebuilding the image:

```bash
docker buildx build \
  --load \
  --tag loom_dev:local \
  .
```

Use `loom_dev:local` in the container commands while testing a local build.

For an offline machine, create and transfer an archive:

```bash
docker save ftod/loom_dev:latest | gzip >loom_dev-latest.tar.gz
gzip -dc loom_dev-latest.tar.gz | docker load
```
