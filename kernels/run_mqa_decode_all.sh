#!/usr/bin/env bash
set -euo pipefail

kernel="mqa_decode"

config_prefix="kernels/config_files/mqa_decode/"
hw_spec_prefix="third_party/loom-mlar/tests/2d_mesh/"

# Run matrix entries are: "kernel_size|config|hw_spec"
# wh
run_matrix_wh=(
  "B8_H32_L1024_D64|fixed_block_size.json|2d_mesh_torus_x2y8.mlir"
  "B8_H32_L2048_D64|fixed_block_size.json|2d_mesh_torus_x4y8.mlir"
  "B8_H32_L4096_D64|fixed_block_size.json|2d_mesh_torus_x8y8.mlir"

  "B16_H32_L1024_D64|fixed_block_size.json|2d_mesh_torus_x2y8.mlir"
  "B16_H32_L2048_D64|fixed_block_size.json|2d_mesh_torus_x4y8.mlir"
  "B16_H32_L4096_D64|fixed_block_size.json|2d_mesh_torus_x8y8.mlir"

  "B24_H32_L1024_D64|fixed_block_size.json|2d_mesh_torus_x2y8.mlir"
  "B24_H32_L2048_D64|fixed_block_size.json|2d_mesh_torus_x4y8.mlir"
  "B24_H32_L4096_D64|fixed_block_size.json|2d_mesh_torus_x8y8.mlir"
)

# bh
run_matrix_bh=(
  "B10_H32_L1024_D64|fixed_block_size.json|2d_mesh_torus_x2y10.mlir"
  "B10_H32_L2048_D64|fixed_block_size.json|2d_mesh_torus_x4y10.mlir"
  "B10_H32_L4096_D64|fixed_block_size.json|2d_mesh_torus_x8y10.mlir"
  
  "B20_H32_L1024_D64|fixed_block_size.json|2d_mesh_torus_x2y10.mlir"
  "B20_H32_L2048_D64|fixed_block_size.json|2d_mesh_torus_x4y10.mlir"
  "B20_H32_L4096_D64|fixed_block_size.json|2d_mesh_torus_x8y10.mlir"

  "B30_H32_L1024_D64|fixed_block_size.json|2d_mesh_torus_x2y10.mlir"
  "B30_H32_L2048_D64|fixed_block_size.json|2d_mesh_torus_x4y10.mlir"
  "B30_H32_L4096_D64|fixed_block_size.json|2d_mesh_torus_x8y10.mlir"
)

for run_item in "${run_matrix_wh[@]}"; do
  IFS="|" read -r kernel_size config_postfix hw_spec_postfix <<< "$run_item"
  config_path="${config_prefix}${config_postfix}"
  hw_spec="${hw_spec_prefix}${hw_spec_postfix}"
  output_path="test/mqa_decode/wh_cap/${kernel_size}"

  python "kernels/${kernel}.py" "-${kernel_size}" \
    --config "$config_path" \
    --output_path "$output_path" \
    --hw_spec "$hw_spec" \
    --njobs 1 \
    --debug
done

for run_item in "${run_matrix_bh[@]}"; do
  IFS="|" read -r kernel_size config_postfix hw_spec_postfix <<< "$run_item"
  config_path="${config_prefix}${config_postfix}"
  hw_spec="${hw_spec_prefix}${hw_spec_postfix}"
  output_path="test/mqa_decode/bh_cap/${kernel_size}"

  python "kernels/${kernel}.py" "-${kernel_size}" \
    --config "$config_path" \
    --output_path "$output_path" \
    --hw_spec "$hw_spec" \
    --njobs 1 \
    --debug
done
