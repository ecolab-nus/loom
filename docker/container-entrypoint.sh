#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" != "sshd" ]; then
    exec "$@"
fi

AUTHORIZED_KEY_FILE="${LOOM_AUTHORIZED_KEY_FILE:-/run/loom/authorized_key}"
if [ ! -s "$AUTHORIZED_KEY_FILE" ]; then
    echo "ERROR: SSH public key not found at $AUTHORIZED_KEY_FILE" >&2
    echo "Mount a public key there as read-only; see docs/docker.md." >&2
    exit 1
fi

install -d -m 0700 /root/.ssh /run/sshd
install -m 0600 "$AUTHORIZED_KEY_FILE" /root/.ssh/authorized_keys
ssh-keygen -A

exec /usr/sbin/sshd -D -e
