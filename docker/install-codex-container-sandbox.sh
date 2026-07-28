#!/usr/bin/env bash

# `sh script.sh` ignores the shebang on Ubuntu and starts dash. Re-exec with
# Bash so the installer behaves the same whether invoked directly or via sh.
if [ -z "${BASH_VERSION:-}" ]; then
  if command -v bash >/dev/null 2>&1; then
    exec bash "$0" "$@"
  fi
  printf '[codex-sandbox] ERROR: Bash is required but was not found.\n' >&2
  exit 1
fi

set -Eeuo pipefail

log() {
  printf '[codex-sandbox] %s\n' "$*"
}

die() {
  printf '[codex-sandbox] ERROR: %s\n' "$*" >&2
  exit 1
}

if [[ "$(uname -s)" != "Linux" ]]; then
  die "This installer is intended for Linux Dev Containers."
fi

if [[ -n "${CODEX_SANDBOX_USER:-}" ]]; then
  TARGET_USER="${CODEX_SANDBOX_USER}"
elif [[ -n "${SUDO_USER:-}" && "$(id -u)" -eq 0 ]]; then
  TARGET_USER="${SUDO_USER}"
elif [[ "$(id -u)" -eq 0 ]]; then
  TARGET_USER="${_REMOTE_USER:-${REMOTE_USER:-${USERNAME:-root}}}"
  if [[ "${TARGET_USER}" == "root" ]]; then
    for vscode_home in /home/*/.vscode-server; do
      if [[ -d "${vscode_home}" ]]; then
        TARGET_USER="$(stat -c '%U' "${vscode_home}")"
        break
      fi
    done
  fi
  if [[ "${TARGET_USER}" == "root" ]] && id ubuntu >/dev/null 2>&1; then
    TARGET_USER="ubuntu"
  fi
else
  TARGET_USER="$(id -un)"
fi

id "${TARGET_USER}" >/dev/null 2>&1 || die "Target user ${TARGET_USER} does not exist."

if [[ "$(id -u)" -eq 0 ]]; then
  AS_ROOT=()
elif command -v sudo >/dev/null 2>&1; then
  AS_ROOT=(sudo)
else
  die "Root access is required. Install sudo or run this script as root."
fi

install_bubblewrap() {
  if command -v bwrap >/dev/null 2>&1; then
    log "bubblewrap is already installed at $(command -v bwrap)"
    return
  fi

  log "Installing bubblewrap in the current container layer..."
  if command -v apt-get >/dev/null 2>&1; then
    "${AS_ROOT[@]}" apt-get update
    "${AS_ROOT[@]}" apt-get install -y --no-install-recommends bubblewrap
  elif command -v dnf >/dev/null 2>&1; then
    "${AS_ROOT[@]}" dnf install -y bubblewrap
  elif command -v yum >/dev/null 2>&1; then
    "${AS_ROOT[@]}" yum install -y bubblewrap
  elif command -v zypper >/dev/null 2>&1; then
    "${AS_ROOT[@]}" zypper --non-interactive install bubblewrap
  elif command -v pacman >/dev/null 2>&1; then
    "${AS_ROOT[@]}" pacman -Sy --noconfirm bubblewrap
  elif command -v apk >/dev/null 2>&1; then
    "${AS_ROOT[@]}" apk add --no-cache bubblewrap
  else
    die "No supported package manager found. Install bubblewrap manually."
  fi
}

install_node_npm() {
  if command -v npm >/dev/null 2>&1; then
    return
  fi

  log "npm is not installed; installing Node.js and npm in the current container layer..."
  if command -v apt-get >/dev/null 2>&1; then
    "${AS_ROOT[@]}" apt-get update
    "${AS_ROOT[@]}" apt-get install -y --no-install-recommends nodejs npm
  elif command -v dnf >/dev/null 2>&1; then
    "${AS_ROOT[@]}" dnf install -y nodejs npm
  elif command -v yum >/dev/null 2>&1; then
    "${AS_ROOT[@]}" yum install -y nodejs npm
  elif command -v zypper >/dev/null 2>&1; then
    "${AS_ROOT[@]}" zypper --non-interactive install nodejs npm
  elif command -v pacman >/dev/null 2>&1; then
    "${AS_ROOT[@]}" pacman -Sy --noconfirm nodejs npm
  elif command -v apk >/dev/null 2>&1; then
    "${AS_ROOT[@]}" apk add --no-cache nodejs npm
  else
    die "No supported package manager found. Install Node.js and npm manually."
  fi

  command -v npm >/dev/null 2>&1 || die "npm installation failed."
}

install_codex_cli() {
  if run_as_target sh -c 'command -v codex >/dev/null 2>&1'; then
    log "Codex CLI is already installed at $(run_as_target sh -c 'command -v codex')"
    return
  fi

  install_node_npm
  log "Installing Codex CLI system-wide with npm..."
  "${AS_ROOT[@]}" npm install --global --prefix /usr/local @openai/codex

  if ! run_as_target sh -c 'command -v codex >/dev/null 2>&1'; then
    die "Codex CLI was installed but is not on ${TARGET_USER}'s PATH."
  fi

  log "Installed $(run_as_target codex --version)"
}

configure_bubblewrap() {
  local bwrap_path
  bwrap_path="$(command -v bwrap)" || die "bubblewrap installation failed."

  log "Enabling setuid mode on ${bwrap_path}..."
  "${AS_ROOT[@]}" chown root:root "${bwrap_path}"
  "${AS_ROOT[@]}" chmod 4755 "${bwrap_path}"

  if [[ ! -u "${bwrap_path}" ]]; then
    die "The setuid bit did not stick. The container filesystem may be mounted with nosuid."
  fi
}

run_as_target() {
  if [[ "$(id -u)" -ne 0 || "${TARGET_USER}" == "root" ]]; then
    "$@"
  elif command -v runuser >/dev/null 2>&1; then
    runuser -u "${TARGET_USER}" -- "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo -u "${TARGET_USER}" -- "$@"
  else
    die "Cannot run the smoke test as container user ${TARGET_USER}; run this installer without sudo."
  fi
}

print_runtime_help() {
  cat >&2 <<'EOF'

The package is installed, but Codex's inner sandbox still cannot start.
Container capabilities/security options cannot be added from inside a running
container. Add the following to .devcontainer/devcontainer.json:

  "runArgs": [
    "--cap-add=SYS_ADMIN",
    "--cap-add=SYS_CHROOT",
    "--cap-add=SETUID",
    "--cap-add=SETGID",
    "--cap-add=SYS_PTRACE",
    "--cap-add=NET_ADMIN",
    "--cap-add=NET_RAW",
    "--security-opt=seccomp=unconfined",
    "--security-opt=apparmor=unconfined"
  ]

Then run "Dev Containers: Rebuild Container" and execute this installer again.
NET_ADMIN is required by bubblewrap's namespace setup on supported setuid
builds. NET_RAW is included to match the Codex secure Dev Container profile.
EOF
}

smoke_test() {
  local test_output

  log "Running sandbox smoke test as ${TARGET_USER}..."

  if run_as_target sh -c 'command -v codex >/dev/null 2>&1'; then
    if test_output="$(run_as_target codex sandbox -- /bin/true 2>&1)"; then
      [[ -z "${test_output}" ]] || printf '%s\n' "${test_output}"
      log "Codex Linux sandbox test passed."
      return
    fi
    printf '%s\n' "${test_output}" >&2
    if [[ "${test_output}" == *"Operation not permitted"* ||
          "${test_output}" == *"Permission denied"* ||
          "${test_output}" == *"bwrap"* ||
          "${test_output}" == *"namespace"* ]]; then
      print_runtime_help
    fi
    die "codex sandbox test failed."
  fi

  log "Codex CLI is not on ${TARGET_USER}'s PATH; skipping the authoritative sandbox test."
  log "Test from the Codex extension by asking it to run: pwd"
}

install_bubblewrap
configure_bubblewrap
install_codex_cli
smoke_test

log "Installation complete for container user ${TARGET_USER}."
log "In the Codex extension, use workspace-write / Ask for approval."
log "This installation lives in the current container layer and must be rerun after the container is rebuilt."
