#!/usr/bin/env bash
set -euo pipefail

kernel="mamba_chunk_scan"
config_prefix="kernels/config_files/"

# Run matrix entries are: "setting|config"
run_matrix=(
  # wormhole
  "L16384_N96_H128_G8_D256_C256|mamba_chunk_scan_wh/L16384_N96_H128_G8_D256_C256.json"
  "L32768_N64_H128_G2_D256_C512|mamba_chunk_scan_wh/L32768_N64_H128_G2_D256_C512.json"
  "L4096_N128_H128_G8_D128_C512|mamba_chunk_scan_wh/L4096_N128_H128_G8_D128_C512.json"
  "L8192_N32_H128_G4_D128_C256|mamba_chunk_scan_wh/L8192_N32_H128_G4_D128_C256.json"
  "L8192_N64_H128_G8_D128_C256|mamba_chunk_scan_wh/L8192_N64_H128_G8_D128_C256.json"
  
  # blackhole
  "L12288_N32_H160_G4_D128_C384|mamba_chunk_scan_bh/L12288_N32_H160_G4_D128_C384.json"
  "L12288_N64_H160_G8_D128_C384|mamba_chunk_scan_bh/L12288_N64_H160_G8_D128_C384.json"
  "L24576_N48_H160_G4_D128_C768|mamba_chunk_scan_bh/L24576_N48_H160_G4_D128_C768.json"
  "L24576_N64_H160_G8_D256_C384|mamba_chunk_scan_bh/L24576_N64_H160_G8_D256_C384.json"
  "L49152_N64_H160_G2_D256_C768|mamba_chunk_scan_bh/L49152_N64_H160_G2_D256_C768.json"

)

for run_item in "${run_matrix[@]}"; do
  IFS="|" read -r setting config_postfix <<< "$run_item"
  config_path="${config_prefix}${config_postfix}"

  python "kernels/${kernel}.py" "-${setting}" \
    --config "$config_path" \
    --njobs 1 \
    --debug
done
