#!/usr/bin/env bash
# Blackhole: sweep mamba chunk-scan shapes through the Loom pipeline + loom2ttkernel lowering,
# then check/benchmark each one with tests/host_chunk_scan.py.
#
# Per shape:
#   1. kernels/mamba_chunk_scan.py   -> test/loom_chunk_scan_bh/<cfg>/IRs/p03_bufferized.mlir
#   2. lower.sh on the best variant  -> test/loom_chunk_scan_bh/<cfg>/kernels/
#   3. copy kernels to tests/kernels/ (host_chunk_scan.py imports host_ttnn.py from there)
#   4. tt-smi reset + tests/host_chunk_scan.py <cfg>
#
# Logs: tmp_logs/loom_chunk_scan_bh/<cfg>.log holds only the host_chunk_scan.py output;
# compile/lowering output goes to <cfg>_compile.log next to it.
#
# The check targets Helion semantics: the causal loop is block-causal with granularity tile_m
# (the solver guarantees tile_k | tile_m), so the host reference runs with --block_size <tile_m>
# read from the generated kernel. The host only accepts block sizes that are multiples of 64; for
# other tile_m the script falls back to --tril_cb, where every block granularity gives the same result.
#
# Env overrides:
#   CARD=0|1        card to use (default 0)
#   NJOBS=16        pipeline workers
#   HOST_ARGS=...   extra host_chunk_scan.py args (default none)
#   GRID_X/GRID_Y   grid reported by the benchmark (default 12x10, the hw_spec mesh)
#   SKIP_COMPILE=1  reuse existing test/loom_chunk_scan_bh/<cfg>/kernels
#   CONFIGS="..."   space-separated subset of shapes to run

set -u

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ARCH=blackhole
CARD="${CARD:-0}"
NJOBS="${NJOBS:-16}"
HOST_ARGS="${HOST_ARGS:-}"
SKIP_COMPILE="${SKIP_COMPILE:-0}"
GRID_X="${GRID_X:-12}"
GRID_Y="${GRID_Y:-10}"
BATCH=2

PYTHON="${REPO_ROOT}/.venv/bin/python"
TT_SMI="${TT_SMI:-/root/.tenstorrent-venv/bin/tt-smi}"
HW_SPEC="third_party/loom-mlar/tests/${ARCH}/2d_mesh_torus.mlir"
OUT_ROOT="${REPO_ROOT}/test/loom_chunk_scan_bh"
LOG_DIR="${REPO_ROOT}/tmp_logs/loom_chunk_scan_bh"
HOST_KERNELS_DIR="${REPO_ROOT}/tests/kernels"

if [[ ! -f "${HW_SPEC}" ]]; then
  echo "${HW_SPEC} not found (run scripts/build-mlar.sh)" >&2
  exit 1
fi

export TT_METAL_HOME="${TT_METAL_HOME:-/opt/tt-mlir/third_party/tt-metal/src/tt-metal}"
export LD_LIBRARY_PATH="/opt/tt-mlir/build/lib:/opt/ttmlir-toolchain/lib:${TT_METAL_HOME}/build/lib:/opt/openmpi-v5.0.7-ulfm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${TT_METAL_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
export TT_VISIBLE_DEVICES="${CARD}"

mkdir -p "${LOG_DIR}" "${HOST_KERNELS_DIR}"

configs=(
L1920_N32_H128_G4_D128_C192
L1920_N32_H128_G8_D128_C192
L3840_N64_H128_G4_D128_C192
L3840_N64_H128_G4_D128_C384
L3840_N64_H128_G8_D128_C192
L3840_N64_H128_G8_D128_C384
L5760_N64_H128_G2_D128_C192
L5760_N64_H128_G2_D128_C384
L5760_N64_H128_G4_D128_C192
L5760_N64_H128_G4_D128_C384
)
if [[ -n "${CONFIGS:-}" ]]; then
  read -r -a configs <<< "${CONFIGS}"
fi

# Write the outer module header plus the first (best) variant's inner module.
# Lowering the full p03 (hundreds of variants) is slow and can hit unrelated failures.
extract_best_variant() {
  local src="$1" dst="$2" end
  end="$(grep -n -m1 '^  }$' "${src}" | cut -d: -f1)"
  [[ -n "${end}" ]] || return 1
  { head -n "${end}" "${src}"; echo "}"; } > "${dst}"
}

compile_cfg() {
  local cfg="$1" out="${OUT_ROOT}/$1"
  mkdir -p "${out}"
  # Drop artifacts of earlier runs so a failed step cannot fall back to stale MLIR/kernels.
  rm -rf "${out}/IRs" "${out}/ttkernel" "${out}/kernels"

  "${PYTHON}" kernels/mamba_chunk_scan.py \
    --config kernels/config_files/mamba_chunk_scan.json \
    --hw-spec "${HW_SPEC}" \
    --output-path "${out}" \
    --kernel-size "B${BATCH}_${cfg}" \
    --topk-candidates 1 \
    --njobs "${NJOBS}" || return 1

  extract_best_variant "${out}/IRs/p03_bufferized.mlir" "${out}/IRs/p03_best.mlir" || return 1

  LOWER_MLIR_OUTPUT_DIR="${out}/ttkernel" \
  SPLIT_KERNEL_OUTPUT_DIR="${out}/kernels" \
  PYTHON="${PYTHON}" \
    ./third_party/loom2ttkernel/lower.sh "${out}/IRs/p03_best.mlir" 1
}

# host_chunk_scan.py args that make its reference match the kernel's Helion semantics.
semantic_args() {
  local tile_m
  tile_m="$(grep -m1 -o '^run = .*' "$1/host_ttnn.py" | grep -o 'tile_m[0-9]*' | grep -o '[0-9]*')"
  [[ -n "${tile_m}" ]] || return 1
  if (( tile_m % 64 == 0 )); then
    echo "--block_size ${tile_m}"
  else
    echo "--tril_cb"
  fi
}

run_on_card() {
  local cfg="$1" args="$2"
  "${TT_SMI}" -r "${CARD}" > /dev/null 2>&1 && \
  timeout 1200s "${PYTHON}" tests/host_chunk_scan.py "${cfg}" \
    --perf_target mlir \
    --grid_x "${GRID_X}" --grid_y "${GRID_Y}" \
    ${args}
}

for cfg in "${configs[@]}"; do
  echo "Running ${cfg}..."
  # The per-shape log holds only host_chunk_scan.py output.
  log="${LOG_DIR}/${cfg}.log"

  if [[ "${SKIP_COMPILE}" != "1" ]]; then
    if ! compile_cfg "${cfg}" > "${LOG_DIR}/${cfg}_compile.log" 2>&1; then
      echo "Compile failed: ${cfg}"
      continue
    fi
  fi

  if ! cp -f "${OUT_ROOT}/${cfg}/kernels/"*.{cpp,py} "${HOST_KERNELS_DIR}/" 2> /dev/null; then
    echo "Missing kernels: ${cfg}"
    continue
  fi

  if ! sem_args="$(semantic_args "${HOST_KERNELS_DIR}")"; then
    echo "Cannot read tile_m from kernels: ${cfg}"
    continue
  fi
  echo "  host args: ${sem_args} ${HOST_ARGS}"

  run_on_card "${cfg}" "${sem_args} ${HOST_ARGS}" > "${log}" 2>&1
  ret=$?

  if [ $ret -eq 124 ]; then
    echo "Timeout: ${cfg}"
  elif [ $ret -ne 0 ]; then
    echo "Failed: ${cfg}"
  else
    echo "Finished: ${cfg}  $(grep -E '^Result:' "${log}" | tail -1)"
  fi
done
