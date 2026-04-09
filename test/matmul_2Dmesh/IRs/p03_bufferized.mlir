module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
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
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f01__n_n_n__is_double_buffer1__tile_k64__tile_m512__tile_n512(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %c131072 = arith.constant 131072 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %20 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %21 = loom.semaphore_take %20 : memref<512x512xf16> -> memref<512x512xf16>
        %22 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %23 = loom.semaphore_take %22 : memref<512x64xf16> -> memref<512x64xf16>
        %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %29 = arith.muli %arg5, %c64 : index
          %30 = arith.muli %arg3, %c131072 : index
          %31 = arith.addi %30, %29 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [512, 64], strides: [256, 1] : memref<4096x256xf16> to memref<512x64xf16, strided<[256, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x64xf16, strided<[256, 1], offset: ?>> to memref<512x64xf16>
          %32 = arith.muli %arg4, %c512 : index
          %33 = arith.muli %arg5, %c262144 : index
          %34 = arith.addi %33, %32 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [64, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
          loom.matmul ins(%23, %25 : memref<512x64xf16>, memref<64x512xf16>) outs(%21 : memref<512x512xf16>)
          loom.semaphore_give %25 : memref<64x512xf16>
          loom.semaphore_give %23 : memref<512x64xf16>
        } {loom.iter_type = #loom.iter_type<sequential>}
        %26 = arith.muli %arg4, %c512 : index
        %27 = arith.muli %arg3, %c2097152 : index
        %28 = arith.addi %27, %26 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%28], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %21, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<512x512xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f01__n_dim_x_level0_bc8_n__is_double_buffer1__tile_k64__tile_m512__tile_n512(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %c131072 = arith.constant 131072 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %20 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %21 = loom.semaphore_take %20 : memref<512x512xf16> -> memref<512x512xf16>
        %22 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %23 = loom.semaphore_take %22 : memref<512x64xf16> -> memref<512x64xf16>
        %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %29 = arith.muli %arg5, %c64 : index
          %30 = arith.muli %arg3, %c131072 : index
          %31 = arith.addi %30, %29 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [512, 64], strides: [256, 1] : memref<4096x256xf16> to memref<512x64xf16, strided<[256, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x64xf16, strided<[256, 1], offset: ?>> to memref<512x64xf16>
          %32 = arith.muli %arg4, %c512 : index
          %33 = arith.muli %arg5, %c262144 : index
          %34 = arith.addi %33, %32 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [64, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
          loom.matmul ins(%23, %25 : memref<512x64xf16>, memref<64x512xf16>) outs(%21 : memref<512x512xf16>)
          loom.semaphore_give %25 : memref<64x512xf16>
          loom.semaphore_give %23 : memref<512x64xf16>
        } {loom.iter_type = #loom.iter_type<sequential>}
        %26 = arith.muli %arg4, %c512 : index
        %27 = arith.muli %arg3, %c2097152 : index
        %28 = arith.addi %27, %26 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%28], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %21, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<512x512xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f01__dim_y_level0_bc8_n_n__is_double_buffer1__tile_k64__tile_m512__tile_n512(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %c131072 = arith.constant 131072 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %20 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %21 = loom.semaphore_take %20 : memref<512x512xf16> -> memref<512x512xf16>
        %22 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %23 = loom.semaphore_take %22 : memref<512x64xf16> -> memref<512x64xf16>
        %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %29 = arith.muli %arg5, %c64 : index
          %30 = arith.muli %arg3, %c131072 : index
          %31 = arith.addi %30, %29 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [512, 64], strides: [256, 1] : memref<4096x256xf16> to memref<512x64xf16, strided<[256, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<512x64xf16, strided<[256, 1], offset: ?>> to memref<512x64xf16>
          %32 = arith.muli %arg4, %c512 : index
          %33 = arith.muli %arg5, %c262144 : index
          %34 = arith.addi %33, %32 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [64, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
          loom.matmul ins(%23, %25 : memref<512x64xf16>, memref<64x512xf16>) outs(%21 : memref<512x512xf16>)
          loom.semaphore_give %25 : memref<64x512xf16>
          loom.semaphore_give %23 : memref<512x64xf16>
        } {loom.iter_type = #loom.iter_type<sequential>}
        %26 = arith.muli %arg4, %c512 : index
        %27 = arith.muli %arg3, %c2097152 : index
        %28 = arith.addi %27, %26 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%28], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %21, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<512x512xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f01__dim_y_level0_bc8_dim_x_level0_bc8_n__is_double_buffer1__tile_k256__tile_m256__tile_n256(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c1048576 = arith.constant 1048576 : index
      %c65536 = arith.constant 65536 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c256 = arith.constant 256 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c2 step %c1 {
          scf.for %arg6 = %c0 to %c2 step %c1 {
            %20 = arith.muli %arg5, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [256, 256] on @L1 : memref<256x256xf16>
            %25 = loom.semaphore_take %24 : memref<256x256xf16> -> memref<256x256xf16>
            %26 = loom.alloc [256, 256] on @L1 : memref<256x256xf16>
            %27 = loom.semaphore_take %26 : memref<256x256xf16> -> memref<256x256xf16>
            %28 = loom.alloc [256, 256] on @L1 : memref<256x256xf16>
            %29 = loom.semaphore_take %28 : memref<256x256xf16> -> memref<256x256xf16>
            %30 = arith.muli %21, %c65536 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [256, 256], strides: [256, 1] : memref<4096x256xf16> to memref<256x256xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<256x256xf16, strided<[256, 1], offset: ?>> to memref<256x256xf16>
            %31 = arith.muli %23, %c256 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [256, 256], strides: [4096, 1] : memref<256x4096xf16> to memref<256x256xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<256x256xf16, strided<[4096, 1], offset: ?>> to memref<256x256xf16>
            loom.matmul ins(%27, %29 : memref<256x256xf16>, memref<256x256xf16>) outs(%25 : memref<256x256xf16>)
            loom.semaphore_give %29 : memref<256x256xf16>
            loom.semaphore_give %27 : memref<256x256xf16>
            %32 = arith.muli %21, %c1048576 : index
            %33 = arith.addi %32, %31 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [256, 256], strides: [4096, 1] : memref<4096x4096xf16> to memref<256x256xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<256x256xf16> to memref<256x256xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<256x256xf16>
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f10__n_n_n__is_double_buffer1__tile_k64__tile_m512__tile_n512(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %c131072 = arith.constant 131072 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %20 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %21 = loom.semaphore_take %20 : memref<512x512xf16> -> memref<512x512xf16>
        %22 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %23 = loom.semaphore_take %22 : memref<512x64xf16> -> memref<512x64xf16>
        %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %29 = arith.muli %arg5, %c64 : index
          %30 = arith.muli %arg3, %c131072 : index
          %31 = arith.addi %30, %29 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [512, 64], strides: [256, 1] : memref<4096x256xf16> to memref<512x64xf16, strided<[256, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x64xf16, strided<[256, 1], offset: ?>> to memref<512x64xf16>
          %32 = arith.muli %arg4, %c512 : index
          %33 = arith.muli %arg5, %c262144 : index
          %34 = arith.addi %33, %32 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [64, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
          loom.matmul ins(%23, %25 : memref<512x64xf16>, memref<64x512xf16>) outs(%21 : memref<512x512xf16>)
          loom.semaphore_give %25 : memref<64x512xf16>
          loom.semaphore_give %23 : memref<512x64xf16>
        } {loom.iter_type = #loom.iter_type<sequential>}
        %26 = arith.muli %arg4, %c512 : index
        %27 = arith.muli %arg3, %c2097152 : index
        %28 = arith.addi %27, %26 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%28], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %21, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<512x512xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f10__n_dim_x_level0_bc8_n__is_double_buffer1__tile_k64__tile_m512__tile_n512(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %c131072 = arith.constant 131072 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %20 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %21 = loom.semaphore_take %20 : memref<512x512xf16> -> memref<512x512xf16>
        %22 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %23 = loom.semaphore_take %22 : memref<512x64xf16> -> memref<512x64xf16>
        %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %29 = arith.muli %arg5, %c64 : index
          %30 = arith.muli %arg3, %c131072 : index
          %31 = arith.addi %30, %29 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [512, 64], strides: [256, 1] : memref<4096x256xf16> to memref<512x64xf16, strided<[256, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x64xf16, strided<[256, 1], offset: ?>> to memref<512x64xf16>
          %32 = arith.muli %arg4, %c512 : index
          %33 = arith.muli %arg5, %c262144 : index
          %34 = arith.addi %33, %32 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [64, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
          loom.matmul ins(%23, %25 : memref<512x64xf16>, memref<64x512xf16>) outs(%21 : memref<512x512xf16>)
          loom.semaphore_give %25 : memref<64x512xf16>
          loom.semaphore_give %23 : memref<512x64xf16>
        } {loom.iter_type = #loom.iter_type<sequential>}
        %26 = arith.muli %arg4, %c512 : index
        %27 = arith.muli %arg3, %c2097152 : index
        %28 = arith.addi %27, %26 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%28], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %21, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<512x512xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f10__dim_y_level0_bc8_n_n__is_double_buffer1__tile_k64__tile_m512__tile_n512(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %c131072 = arith.constant 131072 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %20 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %21 = loom.semaphore_take %20 : memref<512x512xf16> -> memref<512x512xf16>
        %22 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %23 = loom.semaphore_take %22 : memref<512x64xf16> -> memref<512x64xf16>
        %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %29 = arith.muli %arg5, %c64 : index
          %30 = arith.muli %arg3, %c131072 : index
          %31 = arith.addi %30, %29 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [512, 64], strides: [256, 1] : memref<4096x256xf16> to memref<512x64xf16, strided<[256, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<512x64xf16, strided<[256, 1], offset: ?>> to memref<512x64xf16>
          %32 = arith.muli %arg4, %c512 : index
          %33 = arith.muli %arg5, %c262144 : index
          %34 = arith.addi %33, %32 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [64, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
          loom.matmul ins(%23, %25 : memref<512x64xf16>, memref<64x512xf16>) outs(%21 : memref<512x512xf16>)
          loom.semaphore_give %25 : memref<64x512xf16>
          loom.semaphore_give %23 : memref<512x64xf16>
        } {loom.iter_type = #loom.iter_type<sequential>}
        %26 = arith.muli %arg4, %c512 : index
        %27 = arith.muli %arg3, %c2097152 : index
        %28 = arith.addi %27, %26 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%28], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %21, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<512x512xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f10__dim_y_level0_bc8_dim_x_level0_bc8_n__is_double_buffer1__tile_k256__tile_m256__tile_n256(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c1048576 = arith.constant 1048576 : index
      %c65536 = arith.constant 65536 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c256 = arith.constant 256 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c2 step %c1 {
          scf.for %arg6 = %c0 to %c2 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg5, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [256, 256] on @L1 : memref<256x256xf16>
            %25 = loom.semaphore_take %24 : memref<256x256xf16> -> memref<256x256xf16>
            %26 = loom.alloc [256, 256] on @L1 : memref<256x256xf16>
            %27 = loom.semaphore_take %26 : memref<256x256xf16> -> memref<256x256xf16>
            %28 = loom.alloc [256, 256] on @L1 : memref<256x256xf16>
            %29 = loom.semaphore_take %28 : memref<256x256xf16> -> memref<256x256xf16>
            %30 = arith.muli %21, %c65536 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [256, 256], strides: [256, 1] : memref<4096x256xf16> to memref<256x256xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<256x256xf16, strided<[256, 1], offset: ?>> to memref<256x256xf16>
            %31 = arith.muli %23, %c256 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [256, 256], strides: [4096, 1] : memref<256x4096xf16> to memref<256x256xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<256x256xf16, strided<[4096, 1], offset: ?>> to memref<256x256xf16>
            loom.matmul ins(%27, %29 : memref<256x256xf16>, memref<256x256xf16>) outs(%25 : memref<256x256xf16>)
            loom.semaphore_give %29 : memref<256x256xf16>
            loom.semaphore_give %27 : memref<256x256xf16>
            %32 = arith.muli %21, %c1048576 : index
            %33 = arith.addi %32, %31 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [256, 256], strides: [4096, 1] : memref<4096x4096xf16> to memref<256x256xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<256x256xf16> to memref<256x256xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<256x256xf16>
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f01__n_n_n__is_double_buffer1__tile_k64__tile_m512__tile_n512(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %c131072 = arith.constant 131072 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %20 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %21 = loom.semaphore_take %20 : memref<512x512xf16> -> memref<512x512xf16>
        %22 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %23 = loom.semaphore_take %22 : memref<512x64xf16> -> memref<512x64xf16>
        %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %29 = arith.muli %arg5, %c64 : index
          %30 = arith.muli %arg3, %c131072 : index
          %31 = arith.addi %30, %29 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [512, 64], strides: [256, 1] : memref<4096x256xf16> to memref<512x64xf16, strided<[256, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x64xf16, strided<[256, 1], offset: ?>> to memref<512x64xf16>
          %32 = arith.muli %arg4, %c512 : index
          %33 = arith.muli %arg5, %c262144 : index
          %34 = arith.addi %33, %32 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [64, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
          loom.matmul ins(%23, %25 : memref<512x64xf16>, memref<64x512xf16>) outs(%21 : memref<512x512xf16>)
          loom.semaphore_give %25 : memref<64x512xf16>
          loom.semaphore_give %23 : memref<512x64xf16>
        } {loom.iter_type = #loom.iter_type<sequential>}
        %26 = arith.muli %arg4, %c512 : index
        %27 = arith.muli %arg3, %c2097152 : index
        %28 = arith.addi %27, %26 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%28], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %21, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<512x512xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f01__n_dim_y_level0_bc8_n__is_double_buffer1__tile_k64__tile_m512__tile_n512(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %c131072 = arith.constant 131072 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %20 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %21 = loom.semaphore_take %20 : memref<512x512xf16> -> memref<512x512xf16>
        %22 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %23 = loom.semaphore_take %22 : memref<512x64xf16> -> memref<512x64xf16>
        %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %29 = arith.muli %arg5, %c64 : index
          %30 = arith.muli %arg3, %c131072 : index
          %31 = arith.addi %30, %29 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [512, 64], strides: [256, 1] : memref<4096x256xf16> to memref<512x64xf16, strided<[256, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x64xf16, strided<[256, 1], offset: ?>> to memref<512x64xf16>
          %32 = arith.muli %arg4, %c512 : index
          %33 = arith.muli %arg5, %c262144 : index
          %34 = arith.addi %33, %32 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [64, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
          loom.matmul ins(%23, %25 : memref<512x64xf16>, memref<64x512xf16>) outs(%21 : memref<512x512xf16>)
          loom.semaphore_give %25 : memref<64x512xf16>
          loom.semaphore_give %23 : memref<512x64xf16>
        } {loom.iter_type = #loom.iter_type<sequential>}
        %26 = arith.muli %arg4, %c512 : index
        %27 = arith.muli %arg3, %c2097152 : index
        %28 = arith.addi %27, %26 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%28], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %21, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<512x512xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f01__dim_x_level0_bc8_n_n__is_double_buffer1__tile_k64__tile_m512__tile_n512(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %c131072 = arith.constant 131072 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %20 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %21 = loom.semaphore_take %20 : memref<512x512xf16> -> memref<512x512xf16>
        %22 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %23 = loom.semaphore_take %22 : memref<512x64xf16> -> memref<512x64xf16>
        %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %29 = arith.muli %arg5, %c64 : index
          %30 = arith.muli %arg3, %c131072 : index
          %31 = arith.addi %30, %29 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [512, 64], strides: [256, 1] : memref<4096x256xf16> to memref<512x64xf16, strided<[256, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<512x64xf16, strided<[256, 1], offset: ?>> to memref<512x64xf16>
          %32 = arith.muli %arg4, %c512 : index
          %33 = arith.muli %arg5, %c262144 : index
          %34 = arith.addi %33, %32 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [64, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
          loom.matmul ins(%23, %25 : memref<512x64xf16>, memref<64x512xf16>) outs(%21 : memref<512x512xf16>)
          loom.semaphore_give %25 : memref<64x512xf16>
          loom.semaphore_give %23 : memref<512x64xf16>
        } {loom.iter_type = #loom.iter_type<sequential>}
        %26 = arith.muli %arg4, %c512 : index
        %27 = arith.muli %arg3, %c2097152 : index
        %28 = arith.addi %27, %26 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%28], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %21, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<512x512xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f01__dim_x_level0_bc8_dim_y_level0_bc8_n__is_double_buffer1__tile_k64__tile_m512__tile_n512(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %c131072 = arith.constant 131072 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %20 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %21 = loom.semaphore_take %20 : memref<512x512xf16> -> memref<512x512xf16>
        %22 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %23 = loom.semaphore_take %22 : memref<512x64xf16> -> memref<512x64xf16>
        %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %29 = arith.muli %arg5, %c64 : index
          %30 = arith.muli %arg3, %c131072 : index
          %31 = arith.addi %30, %29 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [512, 64], strides: [256, 1] : memref<4096x256xf16> to memref<512x64xf16, strided<[256, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<512x64xf16, strided<[256, 1], offset: ?>> to memref<512x64xf16>
          %32 = arith.muli %arg4, %c512 : index
          %33 = arith.muli %arg5, %c262144 : index
          %34 = arith.addi %33, %32 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [64, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
          loom.matmul ins(%23, %25 : memref<512x64xf16>, memref<64x512xf16>) outs(%21 : memref<512x512xf16>)
          loom.semaphore_give %25 : memref<64x512xf16>
          loom.semaphore_give %23 : memref<512x64xf16>
        } {loom.iter_type = #loom.iter_type<sequential>}
        %26 = arith.muli %arg4, %c512 : index
        %27 = arith.muli %arg3, %c2097152 : index
        %28 = arith.addi %27, %26 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%28], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %21, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<512x512xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f10__n_n_n__is_double_buffer1__tile_k64__tile_m512__tile_n512(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %c131072 = arith.constant 131072 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %20 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %21 = loom.semaphore_take %20 : memref<512x512xf16> -> memref<512x512xf16>
        %22 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %23 = loom.semaphore_take %22 : memref<512x64xf16> -> memref<512x64xf16>
        %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %29 = arith.muli %arg5, %c64 : index
          %30 = arith.muli %arg3, %c131072 : index
          %31 = arith.addi %30, %29 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [512, 64], strides: [256, 1] : memref<4096x256xf16> to memref<512x64xf16, strided<[256, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x64xf16, strided<[256, 1], offset: ?>> to memref<512x64xf16>
          %32 = arith.muli %arg4, %c512 : index
          %33 = arith.muli %arg5, %c262144 : index
          %34 = arith.addi %33, %32 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [64, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
          loom.matmul ins(%23, %25 : memref<512x64xf16>, memref<64x512xf16>) outs(%21 : memref<512x512xf16>)
          loom.semaphore_give %25 : memref<64x512xf16>
          loom.semaphore_give %23 : memref<512x64xf16>
        } {loom.iter_type = #loom.iter_type<sequential>}
        %26 = arith.muli %arg4, %c512 : index
        %27 = arith.muli %arg3, %c2097152 : index
        %28 = arith.addi %27, %26 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%28], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %21, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<512x512xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f10__n_dim_y_level0_bc8_n__is_double_buffer1__tile_k64__tile_m512__tile_n512(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %c131072 = arith.constant 131072 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %20 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %21 = loom.semaphore_take %20 : memref<512x512xf16> -> memref<512x512xf16>
        %22 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %23 = loom.semaphore_take %22 : memref<512x64xf16> -> memref<512x64xf16>
        %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %29 = arith.muli %arg5, %c64 : index
          %30 = arith.muli %arg3, %c131072 : index
          %31 = arith.addi %30, %29 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [512, 64], strides: [256, 1] : memref<4096x256xf16> to memref<512x64xf16, strided<[256, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x64xf16, strided<[256, 1], offset: ?>> to memref<512x64xf16>
          %32 = arith.muli %arg4, %c512 : index
          %33 = arith.muli %arg5, %c262144 : index
          %34 = arith.addi %33, %32 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [64, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
          loom.matmul ins(%23, %25 : memref<512x64xf16>, memref<64x512xf16>) outs(%21 : memref<512x512xf16>)
          loom.semaphore_give %25 : memref<64x512xf16>
          loom.semaphore_give %23 : memref<512x64xf16>
        } {loom.iter_type = #loom.iter_type<sequential>}
        %26 = arith.muli %arg4, %c512 : index
        %27 = arith.muli %arg3, %c2097152 : index
        %28 = arith.addi %27, %26 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%28], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %21, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<512x512xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f10__dim_x_level0_bc8_n_n__is_double_buffer1__tile_k64__tile_m512__tile_n512(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %c131072 = arith.constant 131072 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %20 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %21 = loom.semaphore_take %20 : memref<512x512xf16> -> memref<512x512xf16>
        %22 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %23 = loom.semaphore_take %22 : memref<512x64xf16> -> memref<512x64xf16>
        %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %29 = arith.muli %arg5, %c64 : index
          %30 = arith.muli %arg3, %c131072 : index
          %31 = arith.addi %30, %29 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [512, 64], strides: [256, 1] : memref<4096x256xf16> to memref<512x64xf16, strided<[256, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<512x64xf16, strided<[256, 1], offset: ?>> to memref<512x64xf16>
          %32 = arith.muli %arg4, %c512 : index
          %33 = arith.muli %arg5, %c262144 : index
          %34 = arith.addi %33, %32 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [64, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
          loom.matmul ins(%23, %25 : memref<512x64xf16>, memref<64x512xf16>) outs(%21 : memref<512x512xf16>)
          loom.semaphore_give %25 : memref<64x512xf16>
          loom.semaphore_give %23 : memref<512x64xf16>
        } {loom.iter_type = #loom.iter_type<sequential>}
        %26 = arith.muli %arg4, %c512 : index
        %27 = arith.muli %arg3, %c2097152 : index
        %28 = arith.addi %27, %26 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%28], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %21, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<512x512xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f10__dim_x_level0_bc8_dim_y_level0_bc8_n__is_double_buffer1__tile_k256__tile_m256__tile_n256(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c1048576 = arith.constant 1048576 : index
      %c65536 = arith.constant 65536 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c256 = arith.constant 256 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c2 step %c1 {
          scf.for %arg6 = %c0 to %c2 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg5, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [256, 256] on @L1 : memref<256x256xf16>
            %25 = loom.semaphore_take %24 : memref<256x256xf16> -> memref<256x256xf16>
            %26 = loom.alloc [256, 256] on @L1 : memref<256x256xf16>
            %27 = loom.semaphore_take %26 : memref<256x256xf16> -> memref<256x256xf16>
            %28 = loom.alloc [256, 256] on @L1 : memref<256x256xf16>
            %29 = loom.semaphore_take %28 : memref<256x256xf16> -> memref<256x256xf16>
            %30 = arith.muli %21, %c65536 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [256, 256], strides: [256, 1] : memref<4096x256xf16> to memref<256x256xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<256x256xf16, strided<[256, 1], offset: ?>> to memref<256x256xf16>
            %31 = arith.muli %23, %c256 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [256, 256], strides: [4096, 1] : memref<256x4096xf16> to memref<256x256xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<256x256xf16, strided<[4096, 1], offset: ?>> to memref<256x256xf16>
            loom.matmul ins(%27, %29 : memref<256x256xf16>, memref<256x256xf16>) outs(%25 : memref<256x256xf16>)
            loom.semaphore_give %29 : memref<256x256xf16>
            loom.semaphore_give %27 : memref<256x256xf16>
            %32 = arith.muli %21, %c1048576 : index
            %33 = arith.addi %32, %31 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [256, 256], strides: [4096, 1] : memref<4096x4096xf16> to memref<256x256xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<256x256xf16> to memref<256x256xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<256x256xf16>
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
}
