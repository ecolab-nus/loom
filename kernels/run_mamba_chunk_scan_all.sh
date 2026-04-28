#!/usr/bin/env bash
set -euo pipefail

kernel="mamba_chunk_scan"
config_prefix="kernels/config_files/"

# Run matrix entries are: "setting|config"
run_matrix=(
  # wormhole
  "L1024_N32_H128_G4_D128_C128|mamba_chunk_scan_wh/L1024_N32_H128_G4_D128_C128.json"
  "L1024_N32_H128_G8_D128_C128|mamba_chunk_scan_wh/L1024_N32_H128_G8_D128_C128.json"
  "L2048_N64_H128_G4_D128_C128|mamba_chunk_scan_wh/L2048_N64_H128_G4_D128_C128.json"
  "L2048_N64_H128_G4_D128_C256|mamba_chunk_scan_wh/L2048_N64_H128_G4_D128_C256.json"
  "L2048_N64_H128_G8_D128_C128|mamba_chunk_scan_wh/L2048_N64_H128_G8_D128_C128.json"
  "L2048_N64_H128_G8_D128_C256|mamba_chunk_scan_wh/L2048_N64_H128_G8_D128_C256.json"
  "L4096_N64_H128_G2_D128_C128|mamba_chunk_scan_wh/L4096_N64_H128_G2_D128_C128.json"
  "L4096_N64_H128_G2_D128_C256|mamba_chunk_scan_wh/L4096_N64_H128_G2_D128_C256.json"
  "L4096_N64_H128_G4_D128_C128|mamba_chunk_scan_wh/L4096_N64_H128_G4_D128_C128.json"
  "L4096_N64_H128_G4_D128_C256|mamba_chunk_scan_wh/L4096_N64_H128_G4_D128_C256.json"
  
  # blackhole
  "L1536_N32_H160_G4_D128_C192|mamba_chunk_scan_bh/L1536_N32_H160_G4_D128_C192.json"
  "L1536_N32_H160_G8_D128_C192|mamba_chunk_scan_bh/L1536_N32_H160_G8_D128_C192.json"
  "L3072_N64_H160_G4_D128_C192|mamba_chunk_scan_bh/L3072_N64_H160_G4_D128_C192.json"
  "L3072_N64_H160_G4_D128_C384|mamba_chunk_scan_bh/L3072_N64_H160_G4_D128_C384.json"
  "L3072_N64_H160_G8_D128_C192|mamba_chunk_scan_bh/L3072_N64_H160_G8_D128_C192.json"
  "L3072_N64_H160_G8_D128_C384|mamba_chunk_scan_bh/L3072_N64_H160_G8_D128_C384.json"
  "L6144_N64_H160_G2_D128_C192|mamba_chunk_scan_bh/L6144_N64_H160_G2_D128_C192.json"
  "L6144_N64_H160_G2_D128_C384|mamba_chunk_scan_bh/L6144_N64_H160_G2_D128_C384.json"
  "L6144_N64_H160_G4_D128_C192|mamba_chunk_scan_bh/L6144_N64_H160_G4_D128_C192.json"
  "L6144_N64_H160_G4_D128_C384|mamba_chunk_scan_bh/L6144_N64_H160_G4_D128_C384.json"

)

for run_item in "${run_matrix[@]}"; do
  IFS="|" read -r setting config_postfix <<< "$run_item"
  config_path="${config_prefix}${config_postfix}"

  python "kernels/${kernel}.py" "-${setting}" \
    --config "$config_path" \
    --njobs 1 \
    --debug
done
