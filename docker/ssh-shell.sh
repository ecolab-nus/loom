#!/usr/bin/env bash

set -euo pipefail

# sshd intentionally creates a minimal environment. Restore only the
# toolchain variables captured by the container entrypoint.
source /run/loom/ssh-environment

if [ "$#" -eq 0 ]; then
    exec /bin/bash -l
fi

exec /bin/bash "$@"
