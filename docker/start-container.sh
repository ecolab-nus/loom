#!/usr/bin/env bash

set -euo pipefail

CONTAINER_NAME="${LOOM_CONTAINER_NAME:-loom-dev}"
VOLUME_NAME="${LOOM_VOLUME_NAME:-loom-workspace}"
SSH_PORT="${LOOM_SSH_PORT:-2222}"
AUTHORIZED_KEYS="${LOOM_AUTHORIZED_KEYS:-${HOME}/.ssh/authorized_keys}"
IMAGE="${LOOM_IMAGE:-ftod/loom_dev:latest}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

command -v docker >/dev/null 2>&1 || fail "docker is not installed"
[ -f "$AUTHORIZED_KEYS" ] || fail "authorized keys not found: $AUTHORIZED_KEYS"
[ -s "$AUTHORIZED_KEYS" ] || fail "authorized keys file is empty: $AUTHORIZED_KEYS"

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    fail "container already exists: $CONTAINER_NAME"
fi

docker pull "$IMAGE"
docker volume create "$VOLUME_NAME" >/dev/null

docker create \
    --name "$CONTAINER_NAME" \
    --hostname "$CONTAINER_NAME" \
    --init \
    --restart unless-stopped \
    --publish "127.0.0.1:${SSH_PORT}:22" \
    --mount "type=volume,src=${VOLUME_NAME},dst=/workspace" \
    "$IMAGE" >/dev/null

if ! docker cp "$AUTHORIZED_KEYS" \
    "${CONTAINER_NAME}:/run/loom/authorized_keys"; then
    docker rm "$CONTAINER_NAME" >/dev/null
    fail "could not copy authorized keys into the container"
fi

docker start "$CONTAINER_NAME" >/dev/null

echo "Loom container started."
echo "Connect locally: ssh -A -p ${SSH_PORT} root@localhost"
