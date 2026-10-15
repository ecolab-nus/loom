#!/usr/bin/env bash
# Wormhole: sweep flashattn shapes through Loom and loom2ttkernel, then
# check correctness and benchmark with tests/host_flashattn.py.
# Default shapes match kernels/config_files/flash_attention/wormhole/*.json;
# Loom searches tiling/mapping using the shared flash_attention.json config.
# Each host loads its own generated kernels through LOOM_KERNELS_DIR.
#
# Artifacts: test/loom_flashattn_wh/<cfg>/{IRs,ttkernel,kernels}/
# Logs: tmp_logs/loom_flashattn_wh/<cfg>.log (host only), <cfg>_compile.log.
# A failed shape does not stop the sweep; any failure gives exit status 1.
#
# Env overrides:
#   CARD=0|1        card to reset/use (default 0)
#   NJOBS=16        pipeline workers
#   HOST_ARGS=...   extra host_flashattn.py args, separated by whitespace
#   GRID_X/GRID_Y   benchmark grid (default 8x8, the hw_spec mesh)
#   SKIP_COMPILE=1  reuse existing per-shape kernels
#   CONFIGS="..."   space-separated shapes (e.g. B1_L16384_H64)
#   PYTHON=...      interpreter (default .venv/bin/python)
#   TT_SMI=...      reset tool (default /root/.tenstorrent-venv/bin/tt-smi)

# Non-causal attention; head dimension is fixed to D=65536/H by the kernel.

set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ARCH=wormhole
CARD="${CARD:-0}"
NJOBS="${NJOBS:-16}"
HOST_ARGS="${HOST_ARGS:-}"
SKIP_COMPILE="${SKIP_COMPILE:-0}"
GRID_X="${GRID_X:-8}"
GRID_Y="${GRID_Y:-8}"

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
TT_SMI="${TT_SMI:-/root/.tenstorrent-venv/bin/tt-smi}"
HW_SPEC="third_party/loom-mlar/tests/${ARCH}/2d_mesh_torus.mlir"
OUT_ROOT="${REPO_ROOT}/test/loom_flashattn_wh"
LOG_DIR="${REPO_ROOT}/tmp_logs/loom_flashattn_wh"

echo "Artifacts: ${OUT_ROOT}"
echo "Logs: ${LOG_DIR}"

if [[ ! -f "${HW_SPEC}" ]]; then
  echo "${HW_SPEC} not found (run scripts/build-mlar.sh)" >&2
  exit 1
fi

export TT_METAL_HOME="${TT_METAL_HOME:-/opt/tt-mlir/third_party/tt-metal/src/tt-metal}"
export LD_LIBRARY_PATH="/opt/tt-mlir/build/lib:/opt/ttmlir-toolchain/lib:${TT_METAL_HOME}/build/lib:/opt/openmpi-v5.0.7-ulfm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${TT_METAL_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
export TT_VISIBLE_DEVICES="${CARD}"

mkdir -p "${LOG_DIR}" || exit 1

configs=(
B1_L16384_H64
B1_L16384_H128
B2_L8192_H64
B2_L8192_H128
B4_L4096_H64
B4_L4096_H128
B8_L2048_H64
B8_L2048_H128
B16_L1024_H64
B16_L1024_H128
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
  mkdir -p "${out}" || return 1
  # Drop artifacts of earlier runs so a failed step cannot fall back to stale MLIR/kernels.
  rm -rf "${out}/IRs" "${out}/ttkernel" "${out}/kernels" || return 1

  "${PYTHON}" kernels/flash_attention.py \
    --config kernels/config_files/flash_attention.json \
    --hw-spec "${HW_SPEC}" \
    --output-path "${out}" \
    --kernel-size "${cfg}" \
    --topk-candidates 1 \
    --njobs "${NJOBS}" || return 1

  extract_best_variant "${out}/IRs/p03_bufferized.mlir" "${out}/IRs/p03_best.mlir" || return 1

  LOWER_MLIR_OUTPUT_DIR="${out}/ttkernel" \
  SPLIT_KERNEL_OUTPUT_DIR="${out}/kernels" \
  PYTHON="${PYTHON}" \
    ./third_party/loom2ttkernel/lower.sh "${out}/IRs/p03_best.mlir" 1
}

run_on_card() {
  local cfg="$1"
  "${TT_SMI}" -r "${CARD}" > /dev/null 2>&1 && \
  LOOM_KERNELS_DIR="${OUT_ROOT}/${cfg}/kernels" \
    timeout --kill-after=10s 1200s "${PYTHON}" tests/host_flashattn.py "${cfg}" \
    --perf_target mlir --raise_on_fail \
    --grid_x "${GRID_X}" --grid_y "${GRID_Y}" \
    "${host_args[@]}"
}

# Split HOST_ARGS on whitespace without pathname expansion (same interface as chunk-scan).
host_args=()
read -r -a host_args <<< "${HOST_ARGS}"
failed=0

for cfg in "${configs[@]}"; do
  if [[ ! "${cfg}" =~ ^B[0-9]+_L[0-9]+_H[0-9]+$ ]]; then
    echo "Invalid shape: ${cfg}" >&2
    failed=1
    continue
  fi
  echo "Running ${cfg}..."
  # The per-shape log holds only host_flashattn.py output.
  log="${LOG_DIR}/${cfg}.log"

  if [[ "${SKIP_COMPILE}" != "1" ]]; then
    if ! compile_cfg "${cfg}" > "${LOG_DIR}/${cfg}_compile.log" 2>&1; then
      echo "Compile failed: ${cfg} (see ${LOG_DIR}/${cfg}_compile.log)"
      failed=1
      continue
    fi
  fi

  if [[ ! -f "${OUT_ROOT}/${cfg}/kernels/host_ttnn.py" ]]; then
    echo "Missing kernels: ${cfg}"
    failed=1
    continue
  fi
  echo "  host args: ${HOST_ARGS}"

  run_on_card "${cfg}" > "${log}" 2>&1
  ret=$?

  if [ $ret -eq 124 ]; then
    echo "Timeout: ${cfg}"
    failed=1
  elif [ $ret -ne 0 ]; then
    echo "Failed: ${cfg} (see ${log})"
    failed=1
  else
    echo "Finished: ${cfg}  $(grep -E '^Result:' "${log}" | tail -1)"
  fi
done

exit "${failed}"
