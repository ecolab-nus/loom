# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=ghcr.io/tenstorrent/tt-mlir/tt-mlir-ci-ubuntu-24-04:latest
ARG UV_VERSION=0.11.32

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-bin

FROM ${BASE_IMAGE} AS ttmlir-builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG TTMLIR_COMMIT=5009f4764a31ff08e7cd838ed5747ae8a368a7e6
ARG TT_METAL_COMMIT=ad07818dd4d704d654359c8392794ef8ea8ceec4
ARG CMAKE_BUILD_PARALLEL_LEVEL=8

ENV DEBIAN_FRONTEND=noninteractive \
    TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain \
    TTMLIR_SOURCE_DIR=/opt/tt-mlir \
    TTMLIR_BUILD_DIR=/opt/tt-mlir/build \
    TTMLIR_VENV_DIR=/opt/ttmlir-toolchain/venv \
    MLIR_DIR=/opt/ttmlir-toolchain/lib/cmake/mlir \
    LLVM_DIR=/opt/ttmlir-toolchain/lib/cmake/llvm \
    TT_METAL_RUNTIME_ROOT=/opt/tt-mlir/third_party/tt-metal/src/tt-metal \
    TT_METAL_HOME=/opt/tt-mlir/third_party/tt-metal/src/tt-metal \
    TT_METAL_BUILD_HOME=/opt/tt-mlir/third_party/tt-metal/src/tt-metal/build

# Keep commit/tag history for tt-mlir's version calculation while deferring
# unrelated file contents through Git's partial-clone filter.
WORKDIR /opt/tt-mlir
RUN git clone --filter=blob:none --no-checkout \
      https://github.com/tenstorrent/tt-mlir.git . \
    && git checkout --detach "${TTMLIR_COMMIT}" \
    && test "$(git rev-parse HEAD)" = "${TTMLIR_COMMIT}"

# The default tt-mlir compiler-only build does not fetch TT-Metal. Keep the
# compatible source tree at the path exported by tt-mlir's env/activate.
RUN mkdir -p "${TT_METAL_HOME}" \
    && git -C "${TT_METAL_HOME}" init . \
    && git -C "${TT_METAL_HOME}" remote add origin https://github.com/tenstorrent/tt-metal.git \
    && git -C "${TT_METAL_HOME}" fetch --depth=1 origin "${TT_METAL_COMMIT}" \
    && git -C "${TT_METAL_HOME}" checkout --detach FETCH_HEAD \
    && test "$(git -C "${TT_METAL_HOME}" rev-parse HEAD)" = "${TT_METAL_COMMIT}"

# The CI base image already contains the matching LLVM/MLIR toolchain and venv.
# env/activate selects them for the configure and build steps.
RUN source env/activate \
    && cmake -G Ninja -B build \
    && cmake --build build --parallel "${CMAKE_BUILD_PARALLEL_LEVEL}" \
    && test -x build/bin/ttmlir-opt

FROM ${BASE_IMAGE} AS final

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG TT_INSTALLER_VERSION=v2.1.0

ENV DEBIAN_FRONTEND=noninteractive \
    TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain \
    TTMLIR_SOURCE_DIR=/opt/tt-mlir \
    TTMLIR_BUILD_DIR=/opt/tt-mlir/build \
    TTMLIR_VENV_DIR=/opt/ttmlir-toolchain/venv \
    MLIR_DIR=/opt/ttmlir-toolchain/lib/cmake/mlir \
    LLVM_DIR=/opt/ttmlir-toolchain/lib/cmake/llvm \
    TT_METAL_RUNTIME_ROOT=/opt/tt-mlir/third_party/tt-metal/src/tt-metal \
    TT_METAL_HOME=/opt/tt-mlir/third_party/tt-metal/src/tt-metal \
    TT_METAL_BUILD_HOME=/opt/tt-mlir/third_party/tt-metal/src/tt-metal/build \
    RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo

# Copy compiler artifacts without retaining packages that the build temporarily
# installed into the builder's toolchain venv (notably CUDA-enabled PyTorch).
COPY --from=ttmlir-builder /opt/tt-mlir /opt/tt-mlir

# Run the requested Tenstorrent installer without making host-level changes
# from inside the image. TT-Metal source itself is installed above because this
# installer otherwise expects to pull a separate Metalium runtime container.
RUN curl --proto '=https' --tlsv1.2 -fsSL --retry 5 \
      "https://github.com/tenstorrent/tt-installer/releases/download/${TT_INSTALLER_VERSION}/install.sh" \
      -o /tmp/tt-installer.sh \
    && chmod +x /tmp/tt-installer.sh \
    && /tmp/tt-installer.sh \
      --install-container-runtime=no \
      --mode-container \
      --mode-non-interactive \
      --python-choice=new-venv \
      --no-install-metalium-container \
      --no-install-metalium-models-container \
      --no-install-inference-server \
      --no-install-studio \
      --update-firmware=off \
      --reboot-option=never \
    && rm -f /tmp/tt-installer.sh \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04's packaged Cargo is too old for loom-mlar's Rust 2024 edition.
RUN curl --proto '=https' --tlsv1.2 -fsSL --retry 5 https://sh.rustup.rs \
      -o /tmp/rustup-init.sh \
    && sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain stable \
    && rm -f /tmp/rustup-init.sh \
    && "${CARGO_HOME}/bin/rustc" --version \
    && "${CARGO_HOME}/bin/cargo" --version

# Copy the pinned uv binaries from Astral's official distroless image.
COPY --from=uv-bin /uv /uvx /usr/local/bin/

# Persist the useful parts of `source env/activate` for normal docker run calls.
ENV PATH=/opt/tt-mlir/build/bin:/opt/ttmlir-toolchain/bin:/opt/ttmlir-toolchain/venv/bin:/root/.tenstorrent-venv/bin:/opt/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH=/opt/tt-mlir/build/python_packages:/opt/tt-mlir/build/runtime/python:/opt/ttmlir-toolchain/python_packages/mlir_core \
    LD_LIBRARY_PATH=/opt/tt-mlir/build/lib:/opt/ttmlir-toolchain/lib:/opt/tt-mlir/third_party/tt-metal/src/tt-metal/build/lib

RUN test -f "${MLIR_DIR}/MLIRConfig.cmake" \
    && test -f "${LLVM_DIR}/LLVMConfig.cmake" \
    && test -f "${TT_METAL_HOME}/CMakeLists.txt" \
    && ttmlir-opt --version \
    && mlir-opt --version \
    && rustc --version \
    && cargo --version \
    && uv --version

# Terminal-only developers connect through SSH using a copy of the Docker
# host account's authorized_keys. Password authentication remains disabled.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-server \
    && passwd -d root \
    && mkdir -p /run/sshd /run/loom \
    && rm -rf /var/lib/apt/lists/*

COPY docker/sshd_config /etc/ssh/sshd_config.d/loom.conf
COPY docker/container-entrypoint.sh /usr/local/bin/loom-container-entrypoint
COPY docker/ssh-shell.sh /usr/local/bin/loom-ssh-shell

RUN chmod +x \
      /usr/local/bin/loom-container-entrypoint \
      /usr/local/bin/loom-ssh-shell \
    && usermod --shell /usr/local/bin/loom-ssh-shell root \
    && ssh-keygen -A \
    && /usr/sbin/sshd -t \
    && rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub

WORKDIR /workspace
EXPOSE 22
ENTRYPOINT ["/usr/local/bin/loom-container-entrypoint"]
CMD ["sleep", "infinity"]
