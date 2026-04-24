module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 32 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 128 : index}} {
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
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 32 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 128 : index}} {
    func.func @_mamba_chunk_scan__x2x2x2_y4y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc4_dim_y_level0_bc4_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc4_dim_y_level1_bc8_n_n__is_double_buffer1__tile_b1__tile_c16__tile_h16__tile_k128__tile_m128__tile_n32(%arg0: memref<2x32x8x256x256xf16>, %arg1: memref<2x64x32x256xf16>, %arg2: memref<2x64x32x256xf16>, %arg3: memref<2x8192x64x128xf16>, %arg4: memref<2x8192x8x128xf16>, %arg5: memref<2x32x64x128x128xf16>, %arg6: memref<64xf16>, %arg7: memref<2x8192x64x128xf16>) {
      %c2048 = arith.constant 2048 : index
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c131072 = arith.constant 131072 : index
      %c67108864 = arith.constant 67108864 : index
      %c65536 = arith.constant 65536 : index
      %c16777216 = arith.constant 16777216 : index
      %c33554432 = arith.constant 33554432 : index
      %c1024 = arith.constant 1024 : index
      %c8388608 = arith.constant 8388608 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c4, %c2, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c2 step %c1 {
          %20 = arith.muli %arg13, %c2 overflow<nsw> : index
          %21 = arith.addi %arg8, %20 : index
          %22 = arith.muli %21, %c16 : index
          %23 = arith.muli %arg9, %c128 : index
          %24 = loom.alloc [128] on @L1 : memref<128xf16>
          %25 = loom.semaphore_take %24 : memref<128xf16> -> memref<128xf16>
          %26 = arith.muli %arg11, %c524288 overflow<nsw> : index
          %27 = arith.muli %21, %c131072 : index
          %28 = arith.addi %26, %27 : index
          %29 = arith.muli %arg12, %c4096 : index
          %30 = arith.addi %28, %29 : index
          %31 = arith.addi %30, %23 : index
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128], strides: [1] : memref<2x64x32x256xf16> to memref<128xf16, strided<[1], offset: ?>>
          %32 = arith.muli %arg12, %c2 : index
          %33 = arith.addi %arg9, %32 : index
          %34 = arith.muli %arg8, %c4 : index
          %35 = arith.addi %33, %34 : index
          %36 = arith.muli %arg11, %c4 : index
          %37 = arith.addi %36, %c3 : index
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%35, %36], LR : [%35, %37]) : memref<128xf16, strided<[1], offset: ?>> to memref<128xf16>
          %38 = loom.alloc [128, 32] on @L1 : memref<128x32xf16>
          %39 = loom.semaphore_take %38 : memref<128x32xf16> -> memref<128x32xf16>
          %40 = loom.semaphore_take %38 : memref<128x32xf16> -> memref<128x32xf16>
          %41 = loom.broadcast ins(%25 : memref<128xf16>) outs(%40 : memref<128x32xf16>) dim(1) -> memref<128x32xf16, strided<[?, ?], offset: ?>>
          %42 = arith.addi %23, %29 : index
          %43 = arith.divui %22, %c8 : index
          %44 = loom.alloc [128, 128] on @L1 : memref<128x128xf16>
          %45 = loom.semaphore_take %44 : memref<128x128xf16> -> memref<128x128xf16>
          %46 = arith.muli %arg11, %c8388608 overflow<nsw> : index
          %47 = arith.muli %42, %c1024 overflow<nsw> : index
          %48 = arith.addi %46, %47 : index
          %49 = arith.muli %43, %c128 overflow<nsw> : index
          %50 = arith.addi %48, %49 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%50], sizes: [128, 128], strides: [1024, 1] : memref<2x8192x8x128xf16> to memref<128x128xf16, strided<[1024, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%35, %36], LR : [%35, %37]) : memref<128x128xf16, strided<[1024, 1], offset: ?>> to memref<128x128xf16>
          %51 = arith.muli %arg10, %c32 : index
          %52 = loom.alloc [128, 32] on @L1 : memref<128x32xf16>
          %53 = loom.semaphore_take %52 : memref<128x32xf16> -> memref<128x32xf16>
          %54 = arith.muli %arg11, %c33554432 overflow<nsw> : index
          %55 = arith.muli %arg12, %c16777216 : index
          %56 = arith.addi %54, %55 : index
          %57 = arith.muli %21, %c262144 : index
          %58 = arith.addi %56, %57 : index
          %59 = arith.addi %58, %51 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%59], sizes: [128, 32], strides: [128, 1] : memref<2x32x64x128x128xf16> to memref<128x32xf16, strided<[128, 1], offset: ?>>
          %60 = arith.addi %32, %34 : index
          %61 = arith.addi %32, %c1 : index
          %62 = arith.addi %61, %34 : index
          %63 = arith.addi %arg10, %36 : index
          loom.copy %reinterpret_cast_1, %53 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%60, %63], LR : [%62, %63]) : memref<128x32xf16, strided<[128, 1], offset: ?>> to memref<128x32xf16>
          %64 = loom.alloc [128, 32] on @L1 : memref<128x32xf16>
          %65 = loom.semaphore_take %64 : memref<128x32xf16> -> memref<128x32xf16>
          %66 = loom.semaphore_take %64 : memref<128x32xf16> -> memref<128x32xf16>
          %67 = loom.semaphore_take %64 : memref<128x32xf16> -> memref<128x32xf16>
          loom.matmul ins(%45, %53 : memref<128x128xf16>, memref<128x32xf16>) outs(%66 : memref<128x32xf16>)
          loom.semaphore_give %53 : memref<128x32xf16>
          loom.semaphore_give %45 : memref<128x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%66, %41 : memref<128x32xf16>, memref<128x32xf16, strided<[?, ?], offset: ?>>) outs(%67 : memref<128x32xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %90 = math.exp %in_5 : f16
            %91 = arith.mulf %in, %90 : f16
            linalg.yield %91 : f16
          }
          loom.semaphore_give %66 : memref<128x32xf16>
          loom.semaphore_give %40 : memref<128x32xf16>
          %68 = arith.addi %arg9, %c1 : index
          %69 = arith.muli %68, %c128 : index
          %70 = arith.ceildivui %69, %c128 : index
          %71 = loom.alloc [32, 128] on @L1 : memref<32x128xf16>
          %72 = loom.semaphore_take %71 : memref<32x128xf16> -> memref<32x128xf16>
          %73 = loom.alloc [32, 128] on @L1 : memref<32x128xf16>
          %74 = loom.semaphore_take %73 : memref<32x128xf16> -> memref<32x128xf16>
          %75 = loom.alloc [128, 128] on @L1 : memref<128x128xf16>
          %76 = loom.semaphore_take %75 : memref<128x128xf16> -> memref<128x128xf16>
          %77 = loom.alloc [128, 32] on @L1 : memref<128x32xf16>
          %78 = loom.semaphore_take %77 : memref<128x32xf16> -> memref<128x32xf16>
          scf.for %arg14 = %c0 to %70 step %c1 {
            %90 = arith.muli %arg14, %c128 : index
            %91 = arith.addi %90, %c128 : index
            %92 = arith.cmpi ult, %91, %69 : index
            %93 = arith.select %92, %91, %69 : index
            %94 = arith.subi %93, %90 : index
            %95 = loom.alloc [128, %94] on @L1 : memref<?x?xf16>
            %96 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
            %97 = arith.muli %arg11, %c16777216 overflow<nsw> : index
            %98 = arith.muli %arg12, %c8388608 : index
            %99 = arith.addi %97, %98 : index
            %100 = arith.muli %43, %c65536 overflow<nsw> : index
            %101 = arith.addi %99, %100 : index
            %102 = arith.muli %arg9, %c32768 : index
            %103 = arith.addi %101, %102 : index
            %104 = arith.addi %103, %90 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%104], sizes: [128, %94], strides: [256, 1] : memref<2x32x8x256x256xf16> to memref<128x?xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %96 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%35, %63], LR : [%35, %63]) : memref<128x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
            %105 = loom.alloc [%94] on @L1 : memref<?xf16>
            %106 = loom.semaphore_take %105 : memref<?xf16> -> memref<?xf16>
            %107 = loom.semaphore_take %105 : memref<?xf16> -> memref<?xf16>
            %108 = arith.addi %30, %90 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%108], sizes: [%94], strides: [1] : memref<2x64x32x256xf16> to memref<?xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_6, %107 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%35, %63], LR : [%35, %63]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
            %109 = loom.broadcast ins(%25 : memref<128xf16>) outs(%39 : memref<128x32xf16>) dim(1) -> memref<128x128xf16, strided<[?, ?], offset: ?>>
            %110 = loom.broadcast ins(%107 : memref<?xf16>) outs(%72 : memref<32x128xf16>) dim(0) -> memref<128x128xf16, strided<[?, ?], offset: ?>>
            loom.semaphore_give %107 : memref<?xf16>
            %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%108], sizes: [%94], strides: [1] : memref<2x64x32x256xf16> to memref<?xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_7, %106 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%35, %63], LR : [%35, %63]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
            %111 = loom.broadcast ins(%106 : memref<?xf16>) outs(%74 : memref<32x128xf16>) dim(0) -> memref<128x128xf16, strided<[?, ?], offset: ?>>
            loom.semaphore_give %106 : memref<?xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%96, %109, %110, %111 : memref<?x?xf16>, memref<128x128xf16, strided<[?, ?], offset: ?>>, memref<128x128xf16, strided<[?, ?], offset: ?>>, memref<128x128xf16, strided<[?, ?], offset: ?>>) outs(%76 : memref<128x128xf16>) {
            ^bb0(%in: f16, %in_9: f16, %in_10: f16, %in_11: f16, %out: f16):
              %119 = arith.subf %in_9, %in_10 : f16
              %120 = math.exp %119 : f16
              %121 = arith.mulf %in, %120 : f16
              %122 = arith.mulf %121, %in_11 : f16
              linalg.yield %122 : f16
            }
            loom.semaphore_give %74 : memref<32x128xf16>
            loom.semaphore_give %72 : memref<32x128xf16>
            loom.semaphore_give %96 : memref<?x?xf16>
            loom.semaphore_give %39 : memref<128x32xf16>
            %112 = arith.addi %90, %29 : index
            %113 = arith.muli %arg11, %c67108864 overflow<nsw> : index
            %114 = arith.muli %112, %c8192 overflow<nsw> : index
            %115 = arith.addi %113, %114 : index
            %116 = arith.muli %21, %c2048 : index
            %117 = arith.addi %115, %116 : index
            %118 = arith.addi %117, %51 : index
            %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%118], sizes: [128, 32], strides: [8192, 1] : memref<2x8192x64x128xf16> to memref<128x32xf16, strided<[8192, 1], offset: ?>>
            loom.copy %reinterpret_cast_8, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%35, %63], LR : [%35, %63]) : memref<128x32xf16, strided<[8192, 1], offset: ?>> to memref<128x32xf16>
            linalg.matmul ins(%76, %78 : memref<128x128xf16>, memref<128x32xf16>) outs(%67 : memref<128x32xf16>)
            loom.semaphore_give %78 : memref<128x32xf16>
            loom.semaphore_give %76 : memref<128x128xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %25 : memref<128xf16>
          %79 = loom.alloc [1] on @L1 : memref<f16>
          %80 = loom.semaphore_take %79 : memref<f16> -> memref<f16>
          %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
          %81 = arith.addi %34, %c3 : index
          loom.copy %reinterpret_cast_2, %80 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%34, %c0], LR : [%81, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
          %82 = loom.alloc [128, 32] on @L1 : memref<128x32xf16>
          %83 = loom.semaphore_take %82 : memref<128x32xf16> -> memref<128x32xf16>
          %84 = arith.muli %arg11, %c67108864 overflow<nsw> : index
          %85 = arith.muli %42, %c8192 overflow<nsw> : index
          %86 = arith.addi %84, %85 : index
          %87 = arith.muli %21, %c2048 : index
          %88 = arith.addi %86, %87 : index
          %89 = arith.addi %88, %51 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%89], sizes: [128, 32], strides: [8192, 1] : memref<2x8192x64x128xf16> to memref<128x32xf16, strided<[8192, 1], offset: ?>>
          loom.copy %reinterpret_cast_3, %83 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%35, %63], LR : [%35, %63]) : memref<128x32xf16, strided<[8192, 1], offset: ?>> to memref<128x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%67, %83, %80 : memref<128x32xf16>, memref<128x32xf16>, memref<f16>) outs(%83 : memref<128x32xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %90 = arith.mulf %in_5, %in_6 : f16
            %91 = arith.addf %in, %90 : f16
            linalg.yield %91 : f16
          }
          loom.semaphore_give %80 : memref<f16>
          loom.semaphore_give %67 : memref<128x32xf16>
          loom.sync ins(%83 : memref<128x32xf16>) outs(%65 : memref<128x32xf16>)
          loom.semaphore_give %83 : memref<128x32xf16>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%89], sizes: [128, 32], strides: [8192, 1] : memref<2x8192x64x128xf16> to memref<128x32xf16, strided<[8192, 1], offset: ?>>
          loom.copy %65, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%35, %63], LR : [%35, %63]) : memref<128x32xf16> to memref<128x32xf16, strided<[8192, 1], offset: ?>>
          loom.semaphore_give %65 : memref<128x32xf16>
        } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_x, @dim_x, @dim_y, @dim_y, @dim_x]}
      return
    }
  }
}
