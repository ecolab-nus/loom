module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 32 : index}, loom.tile_h = {is_reduction = false, upper_bound = 32 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 128 : index}} {
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
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 32 : index}, loom.tile_h = {is_reduction = false, upper_bound = 32 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 128 : index}} {
    func.func @_mamba_chunk_scan__x2x2x2_y4y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc4_dim_y_level0_bc4_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc4_dim_y_level1_bc8_n_n__is_double_buffer1__tile_b1__tile_c16__tile_h32__tile_k128__tile_m128__tile_n32(%arg0: memref<2x32x4x256x256xf16>, %arg1: memref<2x32x32x256xf16>, %arg2: memref<2x32x32x256xf16>, %arg3: memref<2x8192x32x128xf16>, %arg4: memref<2x8192x4x128xf16>, %arg5: memref<2x32x32x128x128xf16>, %arg6: memref<32xf16>, %arg7: memref<2x8192x32x128xf16>) {
      %c32768 = arith.constant 32768 : index
      %c524288 = arith.constant 524288 : index
      %c33554432 = arith.constant 33554432 : index
      %c65536 = arith.constant 65536 : index
      %c8388608 = arith.constant 8388608 : index
      %c16777216 = arith.constant 16777216 : index
      %c512 = arith.constant 512 : index
      %c4194304 = arith.constant 4194304 : index
      %c262144 = arith.constant 262144 : index
      %c4096 = arith.constant 4096 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c2 = arith.constant 2 : index
      %c128 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c4, %c2, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        %20 = arith.muli %arg8, %c32 : index
        %21 = arith.muli %arg9, %c128 : index
        %22 = loom.alloc [128] on @L1 : memref<128xf16>
        %23 = loom.semaphore_take %22 : memref<128xf16> -> memref<128xf16>
        %24 = arith.muli %arg11, %c262144 overflow<nsw> : index
        %25 = arith.muli %arg8, %c262144 : index
        %26 = arith.addi %24, %25 : index
        %27 = arith.muli %arg12, %c4096 : index
        %28 = arith.addi %26, %27 : index
        %29 = arith.addi %28, %21 : index
        %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128], strides: [1] : memref<2x32x32x256xf16> to memref<128xf16, strided<[1], offset: ?>>
        %30 = arith.muli %arg12, %c2 : index
        %31 = arith.addi %arg9, %30 : index
        %32 = arith.muli %arg8, %c4 : index
        %33 = arith.addi %31, %32 : index
        %34 = arith.muli %arg11, %c4 : index
        %35 = arith.addi %34, %c3 : index
        loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%33, %34], LR : [%33, %35]) : memref<128xf16, strided<[1], offset: ?>> to memref<128xf16>
        %36 = loom.alloc [128, 32] on @L1 : memref<128x32xf16>
        %37 = loom.semaphore_take %36 : memref<128x32xf16> -> memref<128x32xf16>
        %38 = loom.semaphore_take %36 : memref<128x32xf16> -> memref<128x32xf16>
        %39 = loom.broadcast ins(%23 : memref<128xf16>) outs(%38 : memref<128x32xf16>) dim(1) -> memref<128x32xf16, strided<[?, ?], offset: ?>>
        %40 = arith.addi %21, %27 : index
        %41 = arith.divui %20, %c8 : index
        %42 = loom.alloc [128, 128] on @L1 : memref<128x128xf16>
        %43 = loom.semaphore_take %42 : memref<128x128xf16> -> memref<128x128xf16>
        %44 = arith.muli %arg11, %c4194304 overflow<nsw> : index
        %45 = arith.muli %40, %c512 overflow<nsw> : index
        %46 = arith.addi %44, %45 : index
        %47 = arith.muli %41, %c128 overflow<nsw> : index
        %48 = arith.addi %46, %47 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%48], sizes: [128, 128], strides: [512, 1] : memref<2x8192x4x128xf16> to memref<128x128xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %43 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%33, %34], LR : [%33, %35]) : memref<128x128xf16, strided<[512, 1], offset: ?>> to memref<128x128xf16>
        %49 = arith.muli %arg10, %c32 : index
        %50 = loom.alloc [128, 32] on @L1 : memref<128x32xf16>
        %51 = loom.semaphore_take %50 : memref<128x32xf16> -> memref<128x32xf16>
        %52 = arith.muli %arg11, %c16777216 overflow<nsw> : index
        %53 = arith.muli %arg12, %c8388608 : index
        %54 = arith.addi %52, %53 : index
        %55 = arith.muli %arg8, %c524288 : index
        %56 = arith.addi %54, %55 : index
        %57 = arith.addi %56, %49 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%57], sizes: [128, 32], strides: [128, 1] : memref<2x32x32x128x128xf16> to memref<128x32xf16, strided<[128, 1], offset: ?>>
        %58 = arith.addi %30, %32 : index
        %59 = arith.addi %30, %c1 : index
        %60 = arith.addi %59, %32 : index
        %61 = arith.addi %arg10, %34 : index
        loom.copy %reinterpret_cast_1, %51 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%58, %61], LR : [%60, %61]) : memref<128x32xf16, strided<[128, 1], offset: ?>> to memref<128x32xf16>
        %62 = loom.alloc [128, 32] on @L1 : memref<128x32xf16>
        %63 = loom.semaphore_take %62 : memref<128x32xf16> -> memref<128x32xf16>
        %64 = loom.semaphore_take %62 : memref<128x32xf16> -> memref<128x32xf16>
        %65 = loom.semaphore_take %62 : memref<128x32xf16> -> memref<128x32xf16>
        loom.matmul ins(%43, %51 : memref<128x128xf16>, memref<128x32xf16>) outs(%64 : memref<128x32xf16>)
        loom.semaphore_give %51 : memref<128x32xf16>
        loom.semaphore_give %43 : memref<128x128xf16>
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%64, %39 : memref<128x32xf16>, memref<128x32xf16, strided<[?, ?], offset: ?>>) outs(%65 : memref<128x32xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %88 = math.exp %in_5 : f16
          %89 = arith.mulf %in, %88 : f16
          linalg.yield %89 : f16
        }
        loom.semaphore_give %64 : memref<128x32xf16>
        loom.semaphore_give %38 : memref<128x32xf16>
        %66 = arith.addi %arg9, %c1 : index
        %67 = arith.muli %66, %c128 : index
        %68 = arith.ceildivui %67, %c128 : index
        %69 = loom.alloc [32, 128] on @L1 : memref<32x128xf16>
        %70 = loom.semaphore_take %69 : memref<32x128xf16> -> memref<32x128xf16>
        %71 = loom.alloc [32, 128] on @L1 : memref<32x128xf16>
        %72 = loom.semaphore_take %71 : memref<32x128xf16> -> memref<32x128xf16>
        %73 = loom.alloc [128, 128] on @L1 : memref<128x128xf16>
        %74 = loom.semaphore_take %73 : memref<128x128xf16> -> memref<128x128xf16>
        %75 = loom.alloc [128, 32] on @L1 : memref<128x32xf16>
        %76 = loom.semaphore_take %75 : memref<128x32xf16> -> memref<128x32xf16>
        scf.for %arg13 = %c0 to %68 step %c1 {
          %88 = arith.muli %arg13, %c128 : index
          %89 = arith.addi %88, %c128 : index
          %90 = arith.cmpi ult, %89, %67 : index
          %91 = arith.select %90, %89, %67 : index
          %92 = arith.subi %91, %88 : index
          %93 = loom.alloc [128, %92] on @L1 : memref<?x?xf16>
          %94 = loom.semaphore_take %93 : memref<?x?xf16> -> memref<?x?xf16>
          %95 = arith.muli %arg11, %c8388608 overflow<nsw> : index
          %96 = arith.muli %arg12, %c4194304 : index
          %97 = arith.addi %95, %96 : index
          %98 = arith.muli %41, %c65536 overflow<nsw> : index
          %99 = arith.addi %97, %98 : index
          %100 = arith.muli %arg9, %c32768 : index
          %101 = arith.addi %99, %100 : index
          %102 = arith.addi %101, %88 : index
          %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%102], sizes: [128, %92], strides: [256, 1] : memref<2x32x4x256x256xf16> to memref<128x?xf16, strided<[256, 1], offset: ?>>
          loom.copy %reinterpret_cast_5, %94 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %61], LR : [%33, %61]) : memref<128x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
          %103 = loom.alloc [%92] on @L1 : memref<?xf16>
          %104 = loom.semaphore_take %103 : memref<?xf16> -> memref<?xf16>
          %105 = loom.semaphore_take %103 : memref<?xf16> -> memref<?xf16>
          %106 = arith.addi %28, %88 : index
          %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%106], sizes: [%92], strides: [1] : memref<2x32x32x256xf16> to memref<?xf16, strided<[1], offset: ?>>
          loom.copy %reinterpret_cast_6, %105 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %61], LR : [%33, %61]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
          %107 = loom.broadcast ins(%23 : memref<128xf16>) outs(%37 : memref<128x32xf16>) dim(1) -> memref<128x128xf16, strided<[?, ?], offset: ?>>
          %108 = loom.broadcast ins(%105 : memref<?xf16>) outs(%70 : memref<32x128xf16>) dim(0) -> memref<128x128xf16, strided<[?, ?], offset: ?>>
          loom.semaphore_give %105 : memref<?xf16>
          %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%106], sizes: [%92], strides: [1] : memref<2x32x32x256xf16> to memref<?xf16, strided<[1], offset: ?>>
          loom.copy %reinterpret_cast_7, %104 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %61], LR : [%33, %61]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
          %109 = loom.broadcast ins(%104 : memref<?xf16>) outs(%72 : memref<32x128xf16>) dim(0) -> memref<128x128xf16, strided<[?, ?], offset: ?>>
          loom.semaphore_give %104 : memref<?xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%94, %107, %108, %109 : memref<?x?xf16>, memref<128x128xf16, strided<[?, ?], offset: ?>>, memref<128x128xf16, strided<[?, ?], offset: ?>>, memref<128x128xf16, strided<[?, ?], offset: ?>>) outs(%74 : memref<128x128xf16>) {
          ^bb0(%in: f16, %in_9: f16, %in_10: f16, %in_11: f16, %out: f16):
            %117 = arith.subf %in_9, %in_10 : f16
            %118 = math.exp %117 : f16
            %119 = arith.mulf %in, %118 : f16
            %120 = arith.mulf %119, %in_11 : f16
            linalg.yield %120 : f16
          }
          loom.semaphore_give %72 : memref<32x128xf16>
          loom.semaphore_give %70 : memref<32x128xf16>
          loom.semaphore_give %94 : memref<?x?xf16>
          loom.semaphore_give %37 : memref<128x32xf16>
          %110 = arith.addi %88, %27 : index
          %111 = arith.muli %arg11, %c33554432 overflow<nsw> : index
          %112 = arith.muli %110, %c4096 overflow<nsw> : index
          %113 = arith.addi %111, %112 : index
          %114 = arith.muli %arg8, %c4096 : index
          %115 = arith.addi %113, %114 : index
          %116 = arith.addi %115, %49 : index
          %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%116], sizes: [128, 32], strides: [4096, 1] : memref<2x8192x32x128xf16> to memref<128x32xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_8, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %61], LR : [%33, %61]) : memref<128x32xf16, strided<[4096, 1], offset: ?>> to memref<128x32xf16>
          linalg.matmul ins(%74, %76 : memref<128x128xf16>, memref<128x32xf16>) outs(%65 : memref<128x32xf16>)
          loom.semaphore_give %76 : memref<128x32xf16>
          loom.semaphore_give %74 : memref<128x128xf16>
        } {loom.iter_type = #loom.iter_type<sequential>}
        loom.semaphore_give %23 : memref<128xf16>
        %77 = loom.alloc [1] on @L1 : memref<f16>
        %78 = loom.semaphore_take %77 : memref<f16> -> memref<f16>
        %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%20], sizes: [], strides: [] : memref<32xf16> to memref<f16, strided<[], offset: ?>>
        %79 = arith.addi %32, %c3 : index
        loom.copy %reinterpret_cast_2, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%32, %c0], LR : [%79, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
        %80 = loom.alloc [128, 32] on @L1 : memref<128x32xf16>
        %81 = loom.semaphore_take %80 : memref<128x32xf16> -> memref<128x32xf16>
        %82 = arith.muli %arg11, %c33554432 overflow<nsw> : index
        %83 = arith.muli %40, %c4096 overflow<nsw> : index
        %84 = arith.addi %82, %83 : index
        %85 = arith.muli %arg8, %c4096 : index
        %86 = arith.addi %84, %85 : index
        %87 = arith.addi %86, %49 : index
        %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%87], sizes: [128, 32], strides: [4096, 1] : memref<2x8192x32x128xf16> to memref<128x32xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_3, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %61], LR : [%33, %61]) : memref<128x32xf16, strided<[4096, 1], offset: ?>> to memref<128x32xf16>
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %81, %78 : memref<128x32xf16>, memref<128x32xf16>, memref<f16>) outs(%81 : memref<128x32xf16>) {
        ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
          %88 = arith.mulf %in_5, %in_6 : f16
          %89 = arith.addf %in, %88 : f16
          linalg.yield %89 : f16
        }
        loom.semaphore_give %78 : memref<f16>
        loom.semaphore_give %65 : memref<128x32xf16>
        loom.sync ins(%81 : memref<128x32xf16>) outs(%63 : memref<128x32xf16>)
        loom.semaphore_give %81 : memref<128x32xf16>
        %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%87], sizes: [128, 32], strides: [4096, 1] : memref<2x8192x32x128xf16> to memref<128x32xf16, strided<[4096, 1], offset: ?>>
        loom.copy %63, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%33, %61], LR : [%33, %61]) : memref<128x32xf16> to memref<128x32xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %63 : memref<128x32xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_x, @dim_x, @dim_y, @dim_y, @dim_x]}
      return
    }
  }
}
