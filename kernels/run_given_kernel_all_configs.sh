#!/usr/bin/env bash
set -euo pipefail

settings_root="${1:-kernels/config_files/mamba_chunk_scan}"
kernel="$(basename "$settings_root")"
kernel_script="kernels/${kernel}.py"

if [[ ! -d "$settings_root" ]]; then
  echo "settings root not found: $settings_root" >&2
  exit 1
fi

if [[ ! -f "$kernel_script" ]]; then
  echo "kernel script not found: $kernel_script" >&2
  exit 1
fi

failures=()
found_any=0

while IFS= read -r config_path; do
  found_any=1
  config_path_abs="$(realpath "$config_path")"
  setting="$(basename "${config_path%.json}")"
  echo "Running kernel=$kernel setting=$setting config=$config_path_abs"

  if ! python "$kernel_script" "-${setting}" \
    --config "$config_path_abs" \
    --njobs 1 \
    --debug; then
    echo "Failed kernel=$kernel setting=$setting config=$config_path_abs" >&2
    failures+=("$config_path_abs")
  fi
done < <(find "$settings_root" -mindepth 2 -type f -name '*.json' | sort)

if [[ ${found_any:-0} -eq 0 ]]; then
  echo "No config files found under: $settings_root" >&2
  exit 1
fi

if [[ ${#failures[@]} -gt 0 ]]; then
  echo "" >&2
  echo "Completed with ${#failures[@]} failures:" >&2
  for failed in "${failures[@]}"; do
    echo "  - $failed" >&2
  done
  exit 1
fi
