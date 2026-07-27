#!/usr/bin/env bash

set -euo pipefail

mkdir -p /run/sshd
ssh-keygen -A

if [ -s /run/loom/authorized_keys ]; then
    install -d -m 0700 /root/.ssh
    install -m 0600 /run/loom/authorized_keys /root/.ssh/authorized_keys
fi

SSH_ENVIRONMENT=/run/loom/ssh-environment
: > "$SSH_ENVIRONMENT"
for variable_name in \
    TTMLIR_TOOLCHAIN_DIR \
    TTMLIR_SOURCE_DIR \
    TTMLIR_BUILD_DIR \
    TTMLIR_VENV_DIR \
    MLIR_DIR \
    LLVM_DIR \
    TT_METAL_RUNTIME_ROOT \
    TT_METAL_HOME \
    TT_METAL_BUILD_HOME \
    RUSTUP_HOME \
    CARGO_HOME \
    PATH \
    PYTHONPATH \
    LD_LIBRARY_PATH; do
    printf 'export %s=%q\n' \
        "$variable_name" "${!variable_name:-}" >> "$SSH_ENVIRONMENT"
done
chmod 0600 "$SSH_ENVIRONMENT"

/usr/sbin/sshd

exec "$@"
