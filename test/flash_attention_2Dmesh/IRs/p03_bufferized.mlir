module attributes {loom.tile_b = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
  %0 = adl.memory.bank "mem_DRAM_bank", {bsize = 8192 : i64, nblk = 196608 : i64}
  %1 = adl.spatial_dim "dim_dram_channel", 8
  %2 = adl.memory.array "mem_DRAM", [%1] of %0
  %3 = adl.resource.exclusive "res_L1_torus_h"
  %4 = adl.resource.exclusive "res_L1_torus_v"
  %5 = adl.memory.bank "mem_bank", {bsize = 16 : i64, nblk = 5856 : i64}
  %6 = adl.spatial_dim "dim_nbank", 16
  %7 = adl.memory.array "mem_L1", [%6] of %5
  %8 = adl.resource.exclusive "res_matrix_lane"
  %9 = adl.resource.exclusive "res_vector_lane"
  %10 = adl.processor.compute @proc_matrix_lane, [(%7, %7)], with [%8]
  %11 = adl.processor.compute @proc_vector_lane, [(%7, %7)], with [%9]
  %12 = adl.arch.compose "arch_core", arch[%10, %11], mem[%7]
  %13 = adl.spatial_dim "dim_x", 8
  %14 = adl.spatial_dim "dim_y", 8
  %15 = adl.arch.scale "arch_mesh", [%13, %14] of %12
  %16 = adl.processor.dmover @proc_dram_l1_mover, [(%2, %7), (%7, %2)], with [%3, %4]
  %17 = adl.processor.dmover @proc_dram_l1_bcst_v, [(%2, %7), (%7, %2)], with [%4]
  %18 = adl.processor.dmover @proc_dram_l1_bcst_h, [(%2, %7), (%7, %2)], with [%3]
  %19 = adl.arch.compose "arch_system", arch[%15, %16, %17, %18], mem[%2]
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_flash__attention__x8_y8__d1i0_d0i1__f01__n_n_n_n__is_double_buffer0__tile_b32__tile_m32__tile_n32(%arg0: memref<256x128x4096xf16>, %arg1: memref<256x4096x128xf16>, %arg2: memref<256x4096x128xf16>, %arg3: memref<256x4096x128xf16>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.000000e+00 : f16
      %cst_2 = arith.constant 0xFC00 : f16
      %cst_3 = arith.constant 1.275630e-01 : f16
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %20 = arith.muli %arg6, %c8 overflow<nsw> : index
          %21 = arith.addi %arg5, %20 : index
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %23 = loom.semaphore_take %22 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %24 = loom.semaphore_take %22 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %21, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<32x32x128xf16>
          %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %29 = loom.semaphore_take %28 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          linalg.fill ins(%cst_0 : f16) outs(%29 : memref<32x32x128xf16>)
          %30 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %31 = loom.semaphore_take %30 : memref<32x32xf16> -> memref<32x32xf16>
          linalg.fill ins(%cst_1 : f16) outs(%31 : memref<32x32xf16>)
          %32 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %33 = loom.semaphore_take %32 : memref<32x32xf16> -> memref<32x32xf16>
          linalg.fill ins(%cst_2 : f16) outs(%33 : memref<32x32xf16>)
          %34 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %35 = loom.semaphore_take %34 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %36 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %37 = loom.semaphore_take %36 : memref<32x32xf16> -> memref<32x32xf16>
          %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
          %40 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %41 = loom.semaphore_take %40 : memref<32x32xf16> -> memref<32x32xf16>
          %42 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf16>
          %43 = loom.semaphore_take %42 : memref<32x128x32xf16> -> memref<32x128x32xf16>
          %44 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf16>
          %45 = loom.semaphore_take %44 : memref<32x32x32xf16> -> memref<32x32x32xf16>
          %46 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %47 = loom.semaphore_take %46 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %48 = arith.muli %arg7, %c32 : index
            %49 = arith.addi %25, %48 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%49], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<256x128x4096xf16> to memref<32x128x32xf16, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %43 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x128x32xf16, strided<[524288, 4096, 1], offset: ?>> to memref<32x128x32xf16>
            loom.batch_matmul ins(%24, %43 : memref<32x32x128xf16>, memref<32x128x32xf16>) outs(%45 : memref<32x32x32xf16>)
            loom.semaphore_give %43 : memref<32x128x32xf16>
            linalg.fill ins(%cst_2 : f16) outs(%37 : memref<32x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%45 : memref<32x32x32xf16>) outs(%37 : memref<32x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %52 = arith.maximumf %in, %out : f16
              linalg.yield %52 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %37 : memref<32x32xf16>, memref<32x32xf16>) outs(%37 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.mulf %in_7, %cst_3 : f16
              %53 = arith.cmpf ogt, %in, %52 : f16
              %54 = arith.select %53, %in, %52 : f16
              linalg.yield %54 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %37 : memref<32x32x32xf16>, memref<32x32xf16>) outs(%45 : memref<32x32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.mulf %in, %cst_3 : f16
              %53 = arith.subf %52, %in_7 : f16
              %54 = math.powf %cst, %53 : f16
              linalg.yield %54 : f16
            }
            linalg.fill ins(%cst_0 : f16) outs(%39 : memref<32x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%45 : memref<32x32x32xf16>) outs(%39 : memref<32x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %52 = arith.addf %in, %out : f16
              linalg.yield %52 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %37 : memref<32x32xf16>, memref<32x32xf16>) outs(%41 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.subf %in, %in_7 : f16
              %53 = math.powf %cst, %52 : f16
              linalg.yield %53 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%31, %41, %39 : memref<32x32xf16>, memref<32x32xf16>, memref<32x32xf16>) outs(%31 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
              %52 = arith.mulf %in, %in_7 : f16
              %53 = arith.addf %52, %in_8 : f16
              linalg.yield %53 : f16
            }
            loom.semaphore_give %39 : memref<32x32xf16>
            %50 = arith.muli %arg7, %c4096 : index
            %51 = arith.addi %25, %50 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%51], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<32x32x128xf16>
            loom.batch_matmul ins(%45, %47 : memref<32x32x32xf16>, memref<32x32x128xf16>) outs(%35 : memref<32x32x128xf16>)
            loom.semaphore_give %47 : memref<32x32x128xf16>
            loom.semaphore_give %45 : memref<32x32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %29, %41 : memref<32x32x128xf16>, memref<32x32x128xf16>, memref<32x32xf16>) outs(%29 : memref<32x32x128xf16>) {
            ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
              %52 = arith.mulf %in_7, %in_8 : f16
              %53 = arith.addf %in, %52 : f16
              linalg.yield %53 : f16
            }
            loom.semaphore_give %41 : memref<32x32xf16>
            loom.semaphore_give %35 : memref<32x32x128xf16>
            linalg.copy ins(%37 : memref<32x32xf16>) outs(%33 : memref<32x32xf16>)
            loom.semaphore_give %37 : memref<32x32xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %33 : memref<32x32xf16>
          loom.semaphore_give %24 : memref<32x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %31 : memref<32x32x128xf16>, memref<32x32xf16>) outs(%23 : memref<32x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %48 = arith.divf %in, %in_5 : f16
            linalg.yield %48 : f16
          }
          loom.semaphore_give %31 : memref<32x32xf16>
          loom.semaphore_give %29 : memref<32x32x128xf16>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %23, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %23 : memref<32x32x128xf16>
        } {loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_flash__attention__x8_y8__d1i0_d0i1__f01__n_n_dim_x_level0_bc8_n__is_double_buffer0__tile_b32__tile_m32__tile_n32(%arg0: memref<256x128x4096xf16>, %arg1: memref<256x4096x128xf16>, %arg2: memref<256x4096x128xf16>, %arg3: memref<256x4096x128xf16>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.000000e+00 : f16
      %cst_2 = arith.constant 0xFC00 : f16
      %cst_3 = arith.constant 1.275630e-01 : f16
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %20 = arith.muli %arg6, %c8 overflow<nsw> : index
          %21 = arith.addi %arg5, %20 : index
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %23 = loom.semaphore_take %22 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %24 = loom.semaphore_take %22 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %21, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<32x32x128xf16>
          %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %29 = loom.semaphore_take %28 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          linalg.fill ins(%cst_0 : f16) outs(%29 : memref<32x32x128xf16>)
          %30 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %31 = loom.semaphore_take %30 : memref<32x32xf16> -> memref<32x32xf16>
          linalg.fill ins(%cst_1 : f16) outs(%31 : memref<32x32xf16>)
          %32 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %33 = loom.semaphore_take %32 : memref<32x32xf16> -> memref<32x32xf16>
          linalg.fill ins(%cst_2 : f16) outs(%33 : memref<32x32xf16>)
          %34 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %35 = loom.semaphore_take %34 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %36 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %37 = loom.semaphore_take %36 : memref<32x32xf16> -> memref<32x32xf16>
          %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
          %40 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %41 = loom.semaphore_take %40 : memref<32x32xf16> -> memref<32x32xf16>
          %42 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf16>
          %43 = loom.semaphore_take %42 : memref<32x128x32xf16> -> memref<32x128x32xf16>
          %44 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf16>
          %45 = loom.semaphore_take %44 : memref<32x32x32xf16> -> memref<32x32x32xf16>
          %46 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %47 = loom.semaphore_take %46 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %48 = arith.muli %arg7, %c32 : index
            %49 = arith.addi %25, %48 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%49], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<256x128x4096xf16> to memref<32x128x32xf16, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %43 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x128x32xf16, strided<[524288, 4096, 1], offset: ?>> to memref<32x128x32xf16>
            loom.batch_matmul ins(%24, %43 : memref<32x32x128xf16>, memref<32x128x32xf16>) outs(%45 : memref<32x32x32xf16>)
            loom.semaphore_give %43 : memref<32x128x32xf16>
            linalg.fill ins(%cst_2 : f16) outs(%37 : memref<32x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%45 : memref<32x32x32xf16>) outs(%37 : memref<32x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %52 = arith.maximumf %in, %out : f16
              linalg.yield %52 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %37 : memref<32x32xf16>, memref<32x32xf16>) outs(%37 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.mulf %in_7, %cst_3 : f16
              %53 = arith.cmpf ogt, %in, %52 : f16
              %54 = arith.select %53, %in, %52 : f16
              linalg.yield %54 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %37 : memref<32x32x32xf16>, memref<32x32xf16>) outs(%45 : memref<32x32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.mulf %in, %cst_3 : f16
              %53 = arith.subf %52, %in_7 : f16
              %54 = math.powf %cst, %53 : f16
              linalg.yield %54 : f16
            }
            linalg.fill ins(%cst_0 : f16) outs(%39 : memref<32x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%45 : memref<32x32x32xf16>) outs(%39 : memref<32x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %52 = arith.addf %in, %out : f16
              linalg.yield %52 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %37 : memref<32x32xf16>, memref<32x32xf16>) outs(%41 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.subf %in, %in_7 : f16
              %53 = math.powf %cst, %52 : f16
              linalg.yield %53 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%31, %41, %39 : memref<32x32xf16>, memref<32x32xf16>, memref<32x32xf16>) outs(%31 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
              %52 = arith.mulf %in, %in_7 : f16
              %53 = arith.addf %52, %in_8 : f16
              linalg.yield %53 : f16
            }
            loom.semaphore_give %39 : memref<32x32xf16>
            %50 = arith.muli %arg7, %c4096 : index
            %51 = arith.addi %25, %50 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%51], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<32x32x128xf16>
            loom.batch_matmul ins(%45, %47 : memref<32x32x32xf16>, memref<32x32x128xf16>) outs(%35 : memref<32x32x128xf16>)
            loom.semaphore_give %47 : memref<32x32x128xf16>
            loom.semaphore_give %45 : memref<32x32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %29, %41 : memref<32x32x128xf16>, memref<32x32x128xf16>, memref<32x32xf16>) outs(%29 : memref<32x32x128xf16>) {
            ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
              %52 = arith.mulf %in_7, %in_8 : f16
              %53 = arith.addf %in, %52 : f16
              linalg.yield %53 : f16
            }
            loom.semaphore_give %41 : memref<32x32xf16>
            loom.semaphore_give %35 : memref<32x32x128xf16>
            linalg.copy ins(%37 : memref<32x32xf16>) outs(%33 : memref<32x32xf16>)
            loom.semaphore_give %37 : memref<32x32xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %33 : memref<32x32xf16>
          loom.semaphore_give %24 : memref<32x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %31 : memref<32x32x128xf16>, memref<32x32xf16>) outs(%23 : memref<32x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %48 = arith.divf %in, %in_5 : f16
            linalg.yield %48 : f16
          }
          loom.semaphore_give %31 : memref<32x32xf16>
          loom.semaphore_give %29 : memref<32x32x128xf16>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %23, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %23 : memref<32x32x128xf16>
        } {loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_flash__attention__x8_y8__d1i0_d0i1__f01__n_dim_x_level0_bc8_n_n__is_double_buffer0__tile_b32__tile_m32__tile_n32(%arg0: memref<256x128x4096xf16>, %arg1: memref<256x4096x128xf16>, %arg2: memref<256x4096x128xf16>, %arg3: memref<256x4096x128xf16>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.000000e+00 : f16
      %cst_2 = arith.constant 0xFC00 : f16
      %cst_3 = arith.constant 1.275630e-01 : f16
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %20 = arith.muli %arg6, %c8 overflow<nsw> : index
          %21 = arith.addi %arg5, %20 : index
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %23 = loom.semaphore_take %22 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %24 = loom.semaphore_take %22 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %21, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<32x32x128xf16>
          %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %29 = loom.semaphore_take %28 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          linalg.fill ins(%cst_0 : f16) outs(%29 : memref<32x32x128xf16>)
          %30 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %31 = loom.semaphore_take %30 : memref<32x32xf16> -> memref<32x32xf16>
          linalg.fill ins(%cst_1 : f16) outs(%31 : memref<32x32xf16>)
          %32 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %33 = loom.semaphore_take %32 : memref<32x32xf16> -> memref<32x32xf16>
          linalg.fill ins(%cst_2 : f16) outs(%33 : memref<32x32xf16>)
          %34 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %35 = loom.semaphore_take %34 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %36 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %37 = loom.semaphore_take %36 : memref<32x32xf16> -> memref<32x32xf16>
          %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
          %40 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %41 = loom.semaphore_take %40 : memref<32x32xf16> -> memref<32x32xf16>
          %42 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf16>
          %43 = loom.semaphore_take %42 : memref<32x128x32xf16> -> memref<32x128x32xf16>
          %44 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf16>
          %45 = loom.semaphore_take %44 : memref<32x32x32xf16> -> memref<32x32x32xf16>
          %46 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %47 = loom.semaphore_take %46 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %48 = arith.muli %arg7, %c32 : index
            %49 = arith.addi %25, %48 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%49], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<256x128x4096xf16> to memref<32x128x32xf16, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %43 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<32x128x32xf16, strided<[524288, 4096, 1], offset: ?>> to memref<32x128x32xf16>
            loom.batch_matmul ins(%24, %43 : memref<32x32x128xf16>, memref<32x128x32xf16>) outs(%45 : memref<32x32x32xf16>)
            loom.semaphore_give %43 : memref<32x128x32xf16>
            linalg.fill ins(%cst_2 : f16) outs(%37 : memref<32x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%45 : memref<32x32x32xf16>) outs(%37 : memref<32x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %52 = arith.maximumf %in, %out : f16
              linalg.yield %52 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %37 : memref<32x32xf16>, memref<32x32xf16>) outs(%37 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.mulf %in_7, %cst_3 : f16
              %53 = arith.cmpf ogt, %in, %52 : f16
              %54 = arith.select %53, %in, %52 : f16
              linalg.yield %54 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %37 : memref<32x32x32xf16>, memref<32x32xf16>) outs(%45 : memref<32x32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.mulf %in, %cst_3 : f16
              %53 = arith.subf %52, %in_7 : f16
              %54 = math.powf %cst, %53 : f16
              linalg.yield %54 : f16
            }
            linalg.fill ins(%cst_0 : f16) outs(%39 : memref<32x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%45 : memref<32x32x32xf16>) outs(%39 : memref<32x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %52 = arith.addf %in, %out : f16
              linalg.yield %52 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %37 : memref<32x32xf16>, memref<32x32xf16>) outs(%41 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.subf %in, %in_7 : f16
              %53 = math.powf %cst, %52 : f16
              linalg.yield %53 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%31, %41, %39 : memref<32x32xf16>, memref<32x32xf16>, memref<32x32xf16>) outs(%31 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
              %52 = arith.mulf %in, %in_7 : f16
              %53 = arith.addf %52, %in_8 : f16
              linalg.yield %53 : f16
            }
            loom.semaphore_give %39 : memref<32x32xf16>
            %50 = arith.muli %arg7, %c4096 : index
            %51 = arith.addi %25, %50 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%51], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<32x32x128xf16>
            loom.batch_matmul ins(%45, %47 : memref<32x32x32xf16>, memref<32x32x128xf16>) outs(%35 : memref<32x32x128xf16>)
            loom.semaphore_give %47 : memref<32x32x128xf16>
            loom.semaphore_give %45 : memref<32x32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %29, %41 : memref<32x32x128xf16>, memref<32x32x128xf16>, memref<32x32xf16>) outs(%29 : memref<32x32x128xf16>) {
            ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
              %52 = arith.mulf %in_7, %in_8 : f16
              %53 = arith.addf %in, %52 : f16
              linalg.yield %53 : f16
            }
            loom.semaphore_give %41 : memref<32x32xf16>
            loom.semaphore_give %35 : memref<32x32x128xf16>
            linalg.copy ins(%37 : memref<32x32xf16>) outs(%33 : memref<32x32xf16>)
            loom.semaphore_give %37 : memref<32x32xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %33 : memref<32x32xf16>
          loom.semaphore_give %24 : memref<32x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %31 : memref<32x32x128xf16>, memref<32x32xf16>) outs(%23 : memref<32x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %48 = arith.divf %in, %in_5 : f16
            linalg.yield %48 : f16
          }
          loom.semaphore_give %31 : memref<32x32xf16>
          loom.semaphore_give %29 : memref<32x32x128xf16>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %23, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %23 : memref<32x32x128xf16>
        } {loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_flash__attention__x8_y8__d1i0_d0i1__f01__n_dim_x_level0_bc8_dim_x_level0_bc8_n__is_double_buffer0__tile_b32__tile_m32__tile_n32(%arg0: memref<256x128x4096xf16>, %arg1: memref<256x4096x128xf16>, %arg2: memref<256x4096x128xf16>, %arg3: memref<256x4096x128xf16>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.000000e+00 : f16
      %cst_2 = arith.constant 0xFC00 : f16
      %cst_3 = arith.constant 1.275630e-01 : f16
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %20 = arith.muli %arg6, %c8 overflow<nsw> : index
          %21 = arith.addi %arg5, %20 : index
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %23 = loom.semaphore_take %22 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %24 = loom.semaphore_take %22 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %21, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<32x32x128xf16>
          %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %29 = loom.semaphore_take %28 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          linalg.fill ins(%cst_0 : f16) outs(%29 : memref<32x32x128xf16>)
          %30 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %31 = loom.semaphore_take %30 : memref<32x32xf16> -> memref<32x32xf16>
          linalg.fill ins(%cst_1 : f16) outs(%31 : memref<32x32xf16>)
          %32 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %33 = loom.semaphore_take %32 : memref<32x32xf16> -> memref<32x32xf16>
          linalg.fill ins(%cst_2 : f16) outs(%33 : memref<32x32xf16>)
          %34 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %35 = loom.semaphore_take %34 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %36 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %37 = loom.semaphore_take %36 : memref<32x32xf16> -> memref<32x32xf16>
          %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
          %40 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %41 = loom.semaphore_take %40 : memref<32x32xf16> -> memref<32x32xf16>
          %42 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf16>
          %43 = loom.semaphore_take %42 : memref<32x128x32xf16> -> memref<32x128x32xf16>
          %44 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf16>
          %45 = loom.semaphore_take %44 : memref<32x32x32xf16> -> memref<32x32x32xf16>
          %46 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %47 = loom.semaphore_take %46 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %48 = arith.muli %arg7, %c32 : index
            %49 = arith.addi %25, %48 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%49], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<256x128x4096xf16> to memref<32x128x32xf16, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %43 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<32x128x32xf16, strided<[524288, 4096, 1], offset: ?>> to memref<32x128x32xf16>
            loom.batch_matmul ins(%24, %43 : memref<32x32x128xf16>, memref<32x128x32xf16>) outs(%45 : memref<32x32x32xf16>)
            loom.semaphore_give %43 : memref<32x128x32xf16>
            linalg.fill ins(%cst_2 : f16) outs(%37 : memref<32x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%45 : memref<32x32x32xf16>) outs(%37 : memref<32x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %52 = arith.maximumf %in, %out : f16
              linalg.yield %52 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %37 : memref<32x32xf16>, memref<32x32xf16>) outs(%37 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.mulf %in_7, %cst_3 : f16
              %53 = arith.cmpf ogt, %in, %52 : f16
              %54 = arith.select %53, %in, %52 : f16
              linalg.yield %54 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %37 : memref<32x32x32xf16>, memref<32x32xf16>) outs(%45 : memref<32x32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.mulf %in, %cst_3 : f16
              %53 = arith.subf %52, %in_7 : f16
              %54 = math.powf %cst, %53 : f16
              linalg.yield %54 : f16
            }
            linalg.fill ins(%cst_0 : f16) outs(%39 : memref<32x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%45 : memref<32x32x32xf16>) outs(%39 : memref<32x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %52 = arith.addf %in, %out : f16
              linalg.yield %52 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %37 : memref<32x32xf16>, memref<32x32xf16>) outs(%41 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.subf %in, %in_7 : f16
              %53 = math.powf %cst, %52 : f16
              linalg.yield %53 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%31, %41, %39 : memref<32x32xf16>, memref<32x32xf16>, memref<32x32xf16>) outs(%31 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
              %52 = arith.mulf %in, %in_7 : f16
              %53 = arith.addf %52, %in_8 : f16
              linalg.yield %53 : f16
            }
            loom.semaphore_give %39 : memref<32x32xf16>
            %50 = arith.muli %arg7, %c4096 : index
            %51 = arith.addi %25, %50 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%51], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<32x32x128xf16>
            loom.batch_matmul ins(%45, %47 : memref<32x32x32xf16>, memref<32x32x128xf16>) outs(%35 : memref<32x32x128xf16>)
            loom.semaphore_give %47 : memref<32x32x128xf16>
            loom.semaphore_give %45 : memref<32x32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %29, %41 : memref<32x32x128xf16>, memref<32x32x128xf16>, memref<32x32xf16>) outs(%29 : memref<32x32x128xf16>) {
            ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
              %52 = arith.mulf %in_7, %in_8 : f16
              %53 = arith.addf %in, %52 : f16
              linalg.yield %53 : f16
            }
            loom.semaphore_give %41 : memref<32x32xf16>
            loom.semaphore_give %35 : memref<32x32x128xf16>
            linalg.copy ins(%37 : memref<32x32xf16>) outs(%33 : memref<32x32xf16>)
            loom.semaphore_give %37 : memref<32x32xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %33 : memref<32x32xf16>
          loom.semaphore_give %24 : memref<32x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %31 : memref<32x32x128xf16>, memref<32x32xf16>) outs(%23 : memref<32x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %48 = arith.divf %in, %in_5 : f16
            linalg.yield %48 : f16
          }
          loom.semaphore_give %31 : memref<32x32xf16>
          loom.semaphore_give %29 : memref<32x32x128xf16>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %23, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %23 : memref<32x32x128xf16>
        } {loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_flash__attention__x8_y8__d1i0_d0i1__f10__n_n_n_n__is_double_buffer0__tile_b32__tile_m32__tile_n32(%arg0: memref<256x128x4096xf16>, %arg1: memref<256x4096x128xf16>, %arg2: memref<256x4096x128xf16>, %arg3: memref<256x4096x128xf16>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.000000e+00 : f16
      %cst_2 = arith.constant 0xFC00 : f16
      %cst_3 = arith.constant 1.275630e-01 : f16
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %20 = arith.muli %arg6, %c8 overflow<nsw> : index
          %21 = arith.addi %arg5, %20 : index
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %23 = loom.semaphore_take %22 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %24 = loom.semaphore_take %22 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %21, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<32x32x128xf16>
          %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %29 = loom.semaphore_take %28 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          linalg.fill ins(%cst_0 : f16) outs(%29 : memref<32x32x128xf16>)
          %30 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %31 = loom.semaphore_take %30 : memref<32x32xf16> -> memref<32x32xf16>
          linalg.fill ins(%cst_1 : f16) outs(%31 : memref<32x32xf16>)
          %32 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %33 = loom.semaphore_take %32 : memref<32x32xf16> -> memref<32x32xf16>
          linalg.fill ins(%cst_2 : f16) outs(%33 : memref<32x32xf16>)
          %34 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %35 = loom.semaphore_take %34 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %36 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %37 = loom.semaphore_take %36 : memref<32x32xf16> -> memref<32x32xf16>
          %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
          %40 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %41 = loom.semaphore_take %40 : memref<32x32xf16> -> memref<32x32xf16>
          %42 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf16>
          %43 = loom.semaphore_take %42 : memref<32x128x32xf16> -> memref<32x128x32xf16>
          %44 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf16>
          %45 = loom.semaphore_take %44 : memref<32x32x32xf16> -> memref<32x32x32xf16>
          %46 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %47 = loom.semaphore_take %46 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %48 = arith.muli %arg7, %c32 : index
            %49 = arith.addi %25, %48 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%49], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<256x128x4096xf16> to memref<32x128x32xf16, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %43 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x128x32xf16, strided<[524288, 4096, 1], offset: ?>> to memref<32x128x32xf16>
            loom.batch_matmul ins(%24, %43 : memref<32x32x128xf16>, memref<32x128x32xf16>) outs(%45 : memref<32x32x32xf16>)
            loom.semaphore_give %43 : memref<32x128x32xf16>
            linalg.fill ins(%cst_2 : f16) outs(%37 : memref<32x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%45 : memref<32x32x32xf16>) outs(%37 : memref<32x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %52 = arith.maximumf %in, %out : f16
              linalg.yield %52 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %37 : memref<32x32xf16>, memref<32x32xf16>) outs(%37 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.mulf %in_7, %cst_3 : f16
              %53 = arith.cmpf ogt, %in, %52 : f16
              %54 = arith.select %53, %in, %52 : f16
              linalg.yield %54 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %37 : memref<32x32x32xf16>, memref<32x32xf16>) outs(%45 : memref<32x32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.mulf %in, %cst_3 : f16
              %53 = arith.subf %52, %in_7 : f16
              %54 = math.powf %cst, %53 : f16
              linalg.yield %54 : f16
            }
            linalg.fill ins(%cst_0 : f16) outs(%39 : memref<32x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%45 : memref<32x32x32xf16>) outs(%39 : memref<32x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %52 = arith.addf %in, %out : f16
              linalg.yield %52 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %37 : memref<32x32xf16>, memref<32x32xf16>) outs(%41 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.subf %in, %in_7 : f16
              %53 = math.powf %cst, %52 : f16
              linalg.yield %53 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%31, %41, %39 : memref<32x32xf16>, memref<32x32xf16>, memref<32x32xf16>) outs(%31 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
              %52 = arith.mulf %in, %in_7 : f16
              %53 = arith.addf %52, %in_8 : f16
              linalg.yield %53 : f16
            }
            loom.semaphore_give %39 : memref<32x32xf16>
            %50 = arith.muli %arg7, %c4096 : index
            %51 = arith.addi %25, %50 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%51], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<32x32x128xf16>
            loom.batch_matmul ins(%45, %47 : memref<32x32x32xf16>, memref<32x32x128xf16>) outs(%35 : memref<32x32x128xf16>)
            loom.semaphore_give %47 : memref<32x32x128xf16>
            loom.semaphore_give %45 : memref<32x32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %29, %41 : memref<32x32x128xf16>, memref<32x32x128xf16>, memref<32x32xf16>) outs(%29 : memref<32x32x128xf16>) {
            ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
              %52 = arith.mulf %in_7, %in_8 : f16
              %53 = arith.addf %in, %52 : f16
              linalg.yield %53 : f16
            }
            loom.semaphore_give %41 : memref<32x32xf16>
            loom.semaphore_give %35 : memref<32x32x128xf16>
            linalg.copy ins(%37 : memref<32x32xf16>) outs(%33 : memref<32x32xf16>)
            loom.semaphore_give %37 : memref<32x32xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %33 : memref<32x32xf16>
          loom.semaphore_give %24 : memref<32x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %31 : memref<32x32x128xf16>, memref<32x32xf16>) outs(%23 : memref<32x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %48 = arith.divf %in, %in_5 : f16
            linalg.yield %48 : f16
          }
          loom.semaphore_give %31 : memref<32x32xf16>
          loom.semaphore_give %29 : memref<32x32x128xf16>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %23, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %23 : memref<32x32x128xf16>
        } {loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_flash__attention__x8_y8__d1i0_d0i1__f10__n_n_dim_x_level0_bc8_n__is_double_buffer0__tile_b32__tile_m32__tile_n32(%arg0: memref<256x128x4096xf16>, %arg1: memref<256x4096x128xf16>, %arg2: memref<256x4096x128xf16>, %arg3: memref<256x4096x128xf16>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.000000e+00 : f16
      %cst_2 = arith.constant 0xFC00 : f16
      %cst_3 = arith.constant 1.275630e-01 : f16
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %20 = arith.muli %arg6, %c8 overflow<nsw> : index
          %21 = arith.addi %arg5, %20 : index
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %23 = loom.semaphore_take %22 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %24 = loom.semaphore_take %22 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %21, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<32x32x128xf16>
          %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %29 = loom.semaphore_take %28 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          linalg.fill ins(%cst_0 : f16) outs(%29 : memref<32x32x128xf16>)
          %30 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %31 = loom.semaphore_take %30 : memref<32x32xf16> -> memref<32x32xf16>
          linalg.fill ins(%cst_1 : f16) outs(%31 : memref<32x32xf16>)
          %32 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %33 = loom.semaphore_take %32 : memref<32x32xf16> -> memref<32x32xf16>
          linalg.fill ins(%cst_2 : f16) outs(%33 : memref<32x32xf16>)
          %34 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %35 = loom.semaphore_take %34 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %36 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %37 = loom.semaphore_take %36 : memref<32x32xf16> -> memref<32x32xf16>
          %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
          %40 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %41 = loom.semaphore_take %40 : memref<32x32xf16> -> memref<32x32xf16>
          %42 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf16>
          %43 = loom.semaphore_take %42 : memref<32x128x32xf16> -> memref<32x128x32xf16>
          %44 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf16>
          %45 = loom.semaphore_take %44 : memref<32x32x32xf16> -> memref<32x32x32xf16>
          %46 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %47 = loom.semaphore_take %46 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %48 = arith.muli %arg7, %c32 : index
            %49 = arith.addi %25, %48 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%49], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<256x128x4096xf16> to memref<32x128x32xf16, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %43 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x128x32xf16, strided<[524288, 4096, 1], offset: ?>> to memref<32x128x32xf16>
            loom.batch_matmul ins(%24, %43 : memref<32x32x128xf16>, memref<32x128x32xf16>) outs(%45 : memref<32x32x32xf16>)
            loom.semaphore_give %43 : memref<32x128x32xf16>
            linalg.fill ins(%cst_2 : f16) outs(%37 : memref<32x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%45 : memref<32x32x32xf16>) outs(%37 : memref<32x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %52 = arith.maximumf %in, %out : f16
              linalg.yield %52 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %37 : memref<32x32xf16>, memref<32x32xf16>) outs(%37 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.mulf %in_7, %cst_3 : f16
              %53 = arith.cmpf ogt, %in, %52 : f16
              %54 = arith.select %53, %in, %52 : f16
              linalg.yield %54 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %37 : memref<32x32x32xf16>, memref<32x32xf16>) outs(%45 : memref<32x32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.mulf %in, %cst_3 : f16
              %53 = arith.subf %52, %in_7 : f16
              %54 = math.powf %cst, %53 : f16
              linalg.yield %54 : f16
            }
            linalg.fill ins(%cst_0 : f16) outs(%39 : memref<32x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%45 : memref<32x32x32xf16>) outs(%39 : memref<32x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %52 = arith.addf %in, %out : f16
              linalg.yield %52 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %37 : memref<32x32xf16>, memref<32x32xf16>) outs(%41 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.subf %in, %in_7 : f16
              %53 = math.powf %cst, %52 : f16
              linalg.yield %53 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%31, %41, %39 : memref<32x32xf16>, memref<32x32xf16>, memref<32x32xf16>) outs(%31 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
              %52 = arith.mulf %in, %in_7 : f16
              %53 = arith.addf %52, %in_8 : f16
              linalg.yield %53 : f16
            }
            loom.semaphore_give %39 : memref<32x32xf16>
            %50 = arith.muli %arg7, %c4096 : index
            %51 = arith.addi %25, %50 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%51], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<32x32x128xf16>
            loom.batch_matmul ins(%45, %47 : memref<32x32x32xf16>, memref<32x32x128xf16>) outs(%35 : memref<32x32x128xf16>)
            loom.semaphore_give %47 : memref<32x32x128xf16>
            loom.semaphore_give %45 : memref<32x32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %29, %41 : memref<32x32x128xf16>, memref<32x32x128xf16>, memref<32x32xf16>) outs(%29 : memref<32x32x128xf16>) {
            ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
              %52 = arith.mulf %in_7, %in_8 : f16
              %53 = arith.addf %in, %52 : f16
              linalg.yield %53 : f16
            }
            loom.semaphore_give %41 : memref<32x32xf16>
            loom.semaphore_give %35 : memref<32x32x128xf16>
            linalg.copy ins(%37 : memref<32x32xf16>) outs(%33 : memref<32x32xf16>)
            loom.semaphore_give %37 : memref<32x32xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %33 : memref<32x32xf16>
          loom.semaphore_give %24 : memref<32x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %31 : memref<32x32x128xf16>, memref<32x32xf16>) outs(%23 : memref<32x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %48 = arith.divf %in, %in_5 : f16
            linalg.yield %48 : f16
          }
          loom.semaphore_give %31 : memref<32x32xf16>
          loom.semaphore_give %29 : memref<32x32x128xf16>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %23, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %23 : memref<32x32x128xf16>
        } {loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_flash__attention__x8_y8__d1i0_d0i1__f10__n_dim_x_level0_bc8_n_n__is_double_buffer0__tile_b32__tile_m32__tile_n32(%arg0: memref<256x128x4096xf16>, %arg1: memref<256x4096x128xf16>, %arg2: memref<256x4096x128xf16>, %arg3: memref<256x4096x128xf16>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.000000e+00 : f16
      %cst_2 = arith.constant 0xFC00 : f16
      %cst_3 = arith.constant 1.275630e-01 : f16
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %20 = arith.muli %arg6, %c8 overflow<nsw> : index
          %21 = arith.addi %arg5, %20 : index
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %23 = loom.semaphore_take %22 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %24 = loom.semaphore_take %22 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %21, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<32x32x128xf16>
          %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %29 = loom.semaphore_take %28 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          linalg.fill ins(%cst_0 : f16) outs(%29 : memref<32x32x128xf16>)
          %30 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %31 = loom.semaphore_take %30 : memref<32x32xf16> -> memref<32x32xf16>
          linalg.fill ins(%cst_1 : f16) outs(%31 : memref<32x32xf16>)
          %32 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %33 = loom.semaphore_take %32 : memref<32x32xf16> -> memref<32x32xf16>
          linalg.fill ins(%cst_2 : f16) outs(%33 : memref<32x32xf16>)
          %34 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %35 = loom.semaphore_take %34 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %36 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %37 = loom.semaphore_take %36 : memref<32x32xf16> -> memref<32x32xf16>
          %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
          %40 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %41 = loom.semaphore_take %40 : memref<32x32xf16> -> memref<32x32xf16>
          %42 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf16>
          %43 = loom.semaphore_take %42 : memref<32x128x32xf16> -> memref<32x128x32xf16>
          %44 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf16>
          %45 = loom.semaphore_take %44 : memref<32x32x32xf16> -> memref<32x32x32xf16>
          %46 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %47 = loom.semaphore_take %46 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %48 = arith.muli %arg7, %c32 : index
            %49 = arith.addi %25, %48 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%49], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<256x128x4096xf16> to memref<32x128x32xf16, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %43 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<32x128x32xf16, strided<[524288, 4096, 1], offset: ?>> to memref<32x128x32xf16>
            loom.batch_matmul ins(%24, %43 : memref<32x32x128xf16>, memref<32x128x32xf16>) outs(%45 : memref<32x32x32xf16>)
            loom.semaphore_give %43 : memref<32x128x32xf16>
            linalg.fill ins(%cst_2 : f16) outs(%37 : memref<32x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%45 : memref<32x32x32xf16>) outs(%37 : memref<32x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %52 = arith.maximumf %in, %out : f16
              linalg.yield %52 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %37 : memref<32x32xf16>, memref<32x32xf16>) outs(%37 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.mulf %in_7, %cst_3 : f16
              %53 = arith.cmpf ogt, %in, %52 : f16
              %54 = arith.select %53, %in, %52 : f16
              linalg.yield %54 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %37 : memref<32x32x32xf16>, memref<32x32xf16>) outs(%45 : memref<32x32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.mulf %in, %cst_3 : f16
              %53 = arith.subf %52, %in_7 : f16
              %54 = math.powf %cst, %53 : f16
              linalg.yield %54 : f16
            }
            linalg.fill ins(%cst_0 : f16) outs(%39 : memref<32x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%45 : memref<32x32x32xf16>) outs(%39 : memref<32x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %52 = arith.addf %in, %out : f16
              linalg.yield %52 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %37 : memref<32x32xf16>, memref<32x32xf16>) outs(%41 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.subf %in, %in_7 : f16
              %53 = math.powf %cst, %52 : f16
              linalg.yield %53 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%31, %41, %39 : memref<32x32xf16>, memref<32x32xf16>, memref<32x32xf16>) outs(%31 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
              %52 = arith.mulf %in, %in_7 : f16
              %53 = arith.addf %52, %in_8 : f16
              linalg.yield %53 : f16
            }
            loom.semaphore_give %39 : memref<32x32xf16>
            %50 = arith.muli %arg7, %c4096 : index
            %51 = arith.addi %25, %50 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%51], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<32x32x128xf16>
            loom.batch_matmul ins(%45, %47 : memref<32x32x32xf16>, memref<32x32x128xf16>) outs(%35 : memref<32x32x128xf16>)
            loom.semaphore_give %47 : memref<32x32x128xf16>
            loom.semaphore_give %45 : memref<32x32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %29, %41 : memref<32x32x128xf16>, memref<32x32x128xf16>, memref<32x32xf16>) outs(%29 : memref<32x32x128xf16>) {
            ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
              %52 = arith.mulf %in_7, %in_8 : f16
              %53 = arith.addf %in, %52 : f16
              linalg.yield %53 : f16
            }
            loom.semaphore_give %41 : memref<32x32xf16>
            loom.semaphore_give %35 : memref<32x32x128xf16>
            linalg.copy ins(%37 : memref<32x32xf16>) outs(%33 : memref<32x32xf16>)
            loom.semaphore_give %37 : memref<32x32xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %33 : memref<32x32xf16>
          loom.semaphore_give %24 : memref<32x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %31 : memref<32x32x128xf16>, memref<32x32xf16>) outs(%23 : memref<32x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %48 = arith.divf %in, %in_5 : f16
            linalg.yield %48 : f16
          }
          loom.semaphore_give %31 : memref<32x32xf16>
          loom.semaphore_give %29 : memref<32x32x128xf16>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %23, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %23 : memref<32x32x128xf16>
        } {loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_flash__attention__x8_y8__d1i0_d0i1__f10__n_dim_x_level0_bc8_dim_x_level0_bc8_n__is_double_buffer0__tile_b32__tile_m32__tile_n32(%arg0: memref<256x128x4096xf16>, %arg1: memref<256x4096x128xf16>, %arg2: memref<256x4096x128xf16>, %arg3: memref<256x4096x128xf16>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.000000e+00 : f16
      %cst_2 = arith.constant 0xFC00 : f16
      %cst_3 = arith.constant 1.275630e-01 : f16
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %20 = arith.muli %arg6, %c8 overflow<nsw> : index
          %21 = arith.addi %arg5, %20 : index
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %23 = loom.semaphore_take %22 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %24 = loom.semaphore_take %22 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %21, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<32x32x128xf16>
          %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %29 = loom.semaphore_take %28 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          linalg.fill ins(%cst_0 : f16) outs(%29 : memref<32x32x128xf16>)
          %30 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %31 = loom.semaphore_take %30 : memref<32x32xf16> -> memref<32x32xf16>
          linalg.fill ins(%cst_1 : f16) outs(%31 : memref<32x32xf16>)
          %32 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %33 = loom.semaphore_take %32 : memref<32x32xf16> -> memref<32x32xf16>
          linalg.fill ins(%cst_2 : f16) outs(%33 : memref<32x32xf16>)
          %34 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %35 = loom.semaphore_take %34 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          %36 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %37 = loom.semaphore_take %36 : memref<32x32xf16> -> memref<32x32xf16>
          %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
          %40 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
          %41 = loom.semaphore_take %40 : memref<32x32xf16> -> memref<32x32xf16>
          %42 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf16>
          %43 = loom.semaphore_take %42 : memref<32x128x32xf16> -> memref<32x128x32xf16>
          %44 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf16>
          %45 = loom.semaphore_take %44 : memref<32x32x32xf16> -> memref<32x32x32xf16>
          %46 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf16>
          %47 = loom.semaphore_take %46 : memref<32x32x128xf16> -> memref<32x32x128xf16>
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %48 = arith.muli %arg7, %c32 : index
            %49 = arith.addi %25, %48 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%49], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<256x128x4096xf16> to memref<32x128x32xf16, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %43 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<32x128x32xf16, strided<[524288, 4096, 1], offset: ?>> to memref<32x128x32xf16>
            loom.batch_matmul ins(%24, %43 : memref<32x32x128xf16>, memref<32x128x32xf16>) outs(%45 : memref<32x32x32xf16>)
            loom.semaphore_give %43 : memref<32x128x32xf16>
            linalg.fill ins(%cst_2 : f16) outs(%37 : memref<32x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%45 : memref<32x32x32xf16>) outs(%37 : memref<32x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %52 = arith.maximumf %in, %out : f16
              linalg.yield %52 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %37 : memref<32x32xf16>, memref<32x32xf16>) outs(%37 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.mulf %in_7, %cst_3 : f16
              %53 = arith.cmpf ogt, %in, %52 : f16
              %54 = arith.select %53, %in, %52 : f16
              linalg.yield %54 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %37 : memref<32x32x32xf16>, memref<32x32xf16>) outs(%45 : memref<32x32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.mulf %in, %cst_3 : f16
              %53 = arith.subf %52, %in_7 : f16
              %54 = math.powf %cst, %53 : f16
              linalg.yield %54 : f16
            }
            linalg.fill ins(%cst_0 : f16) outs(%39 : memref<32x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%45 : memref<32x32x32xf16>) outs(%39 : memref<32x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %52 = arith.addf %in, %out : f16
              linalg.yield %52 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %37 : memref<32x32xf16>, memref<32x32xf16>) outs(%41 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %out: f16):
              %52 = arith.subf %in, %in_7 : f16
              %53 = math.powf %cst, %52 : f16
              linalg.yield %53 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%31, %41, %39 : memref<32x32xf16>, memref<32x32xf16>, memref<32x32xf16>) outs(%31 : memref<32x32xf16>) {
            ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
              %52 = arith.mulf %in, %in_7 : f16
              %53 = arith.addf %52, %in_8 : f16
              linalg.yield %53 : f16
            }
            loom.semaphore_give %39 : memref<32x32xf16>
            %50 = arith.muli %arg7, %c4096 : index
            %51 = arith.addi %25, %50 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%51], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<32x32x128xf16>
            loom.batch_matmul ins(%45, %47 : memref<32x32x32xf16>, memref<32x32x128xf16>) outs(%35 : memref<32x32x128xf16>)
            loom.semaphore_give %47 : memref<32x32x128xf16>
            loom.semaphore_give %45 : memref<32x32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %29, %41 : memref<32x32x128xf16>, memref<32x32x128xf16>, memref<32x32xf16>) outs(%29 : memref<32x32x128xf16>) {
            ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
              %52 = arith.mulf %in_7, %in_8 : f16
              %53 = arith.addf %in, %52 : f16
              linalg.yield %53 : f16
            }
            loom.semaphore_give %41 : memref<32x32xf16>
            loom.semaphore_give %35 : memref<32x32x128xf16>
            linalg.copy ins(%37 : memref<32x32xf16>) outs(%33 : memref<32x32xf16>)
            loom.semaphore_give %37 : memref<32x32xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %33 : memref<32x32xf16>
          loom.semaphore_give %24 : memref<32x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %31 : memref<32x32x128xf16>, memref<32x32xf16>) outs(%23 : memref<32x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %48 = arith.divf %in, %in_5 : f16
            linalg.yield %48 : f16
          }
          loom.semaphore_give %31 : memref<32x32xf16>
          loom.semaphore_give %29 : memref<32x32x128xf16>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<256x4096x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %23, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32x128xf16> to memref<32x32x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %23 : memref<32x32x128xf16>
        } {loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
}
