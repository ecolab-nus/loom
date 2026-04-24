module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 128 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 128 : index}} {
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
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 128 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 128 : index}} {
    func.func @_mamba_chunk_scan__x2x4_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc8_dim_y_level1_bc4_n_n(%arg0: memref<2x8x8x512x512xf16>, %arg1: memref<2x128x8x512xf16>, %arg2: memref<2x128x8x512xf16>, %arg3: memref<2x4096x128x128xf16>, %arg4: memref<2x4096x8x128xf16>, %arg5: memref<2x8x128x128x128xf16>, %arg6: memref<128xf16>, %arg7: memref<2x4096x128x128xf16>) {
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c512 = arith.constant 512 : index
      %c128 = arith.constant 128 : index
      %20 = loom.sym @tile_m {upper_bound = 512 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 128 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 128 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c128, %23 : index
      %27 = arith.ceildivui %c512, %20 : index
      %28 = arith.ceildivui %c128, %21 : index
      %29 = arith.ceildivui %c2, %24 : index
      %30 = arith.ceildivui %c8, %25 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (4) {
              affine.parallel (%arg12) = (0) to (2) {
                %31 = arith.ceildivui %26, %c2 : index
                scf.for %arg13 = %c0 to %31 step %c1 {
                  %32 = arith.ceildivui %27, %c2 : index
                  scf.for %arg14 = %c0 to %32 step %c1 {
                    %33 = arith.ceildivui %28, %c2 : index
                    scf.for %arg15 = %c0 to %33 step %c1 {
                      %34 = arith.ceildivui %29, %c4 : index
                      scf.for %arg16 = %c0 to %34 step %c1 {
                        %35 = arith.ceildivui %30, %c2 : index
                        scf.for %arg17 = %c0 to %35 step %c1 {
                          %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                          %37 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                          %38 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg10, %arg15)
                          %39 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg11, %arg16)
                          %40 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg12, %arg17)
                          %41 = arith.muli %39, %24 : index
                          %42 = arith.muli %36, %23 : index
                          %43 = arith.muli %40, %25 : index
                          %44 = arith.muli %37, %20 : index
                          %45 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %46 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg11, %c2 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg12, %c2 : index
                          %51 = arith.muli %arg8, %c4 : index
                          %52 = arith.addi %50, %51 : index
                          %53 = arith.addi %50, %c1 : index
                          %54 = arith.addi %53, %51 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %55 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %56 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %57 = loom.semaphore_take %56 : memref<?x32xf16> -> memref<?x32xf16>
                          %58 = loom.init_tensor %57[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %59 = loom.semaphore_take %56 : memref<?x32xf16> -> memref<?x32xf16>
                          %60 = loom.init_tensor %59[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %61 = loom.broadcast ins(%55 : tensor<?xf16>) outs(%60 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %62 = arith.muli %43, %c512 : index
                          %63 = arith.addi %44, %62 : index
                          %64 = arith.divui %42, %c16 : index
                          %65 = loom.alloc [%20, 128] on @L1 : memref<?x128xf16>
                          %66 = loom.semaphore_take %65 : memref<?x128xf16> -> memref<?x128xf16>
                          %67 = loom.subview %arg4[%41, %63, %64, 0] [1, %20, 1, 128] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x4096x8x128xf16> to memref<?x128xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %67, %66 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?x128xf16, strided<[1024, 1], offset: ?>> to memref<?x128xf16>
                          %68 = loom.bufferize_to_tensor %66[%20, 128] : memref<?x128xf16> -> tensor<?x128xf16>
                          %69 = arith.muli %38, %21 : index
                          %70 = loom.alloc [128, %21] on @L1 : memref<128x?xf16>
                          %71 = loom.semaphore_take %70 : memref<128x?xf16> -> memref<128x?xf16>
                          %72 = loom.subview %arg5[%41, %43, %42, 0, %69] [1, 1, 1, 128, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x128x128x128xf16> to memref<128x?xf16, strided<[128, 1], offset: ?>>
                          %73 = arith.addi %48, %c1 : index
                          %74 = arith.addi %arg10, %50 : index
                          %75 = arith.addi %74, %51 : index
                          loom.copy %72, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%48, %75], LR : [%73, %75]) : memref<128x?xf16, strided<[128, 1], offset: ?>> to memref<128x?xf16>
                          %76 = loom.bufferize_to_tensor %71[128, %21] : memref<128x?xf16> -> tensor<128x?xf16>
                          %77 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %79 = loom.init_tensor %78[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %80 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %81 = loom.init_tensor %80[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %82 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %83 = loom.init_tensor %82[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %84 = linalg.fill ins(%cst : f16) outs(%81 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %85 = linalg.matmul ins(%68, %76 : tensor<?x128xf16>, tensor<128x?xf16>) outs(%84 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %71 : memref<128x?xf16>
                          loom.semaphore_give %66 : memref<?x128xf16>
                          %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%85, %61 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%83 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %116 = math.exp %in_0 : f16
                            %117 = arith.mulf %in, %116 : f16
                            linalg.yield %117 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %80 : memref<?x?xf16>
                          loom.semaphore_give %59 : memref<?x32xf16>
                          %87 = arith.addi %37, %c1 : index
                          %88 = arith.muli %87, %20 : index
                          %89 = arith.ceildivui %88, %22 : index
                          %90 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %91 = loom.semaphore_take %90 : memref<32x?xf16> -> memref<32x?xf16>
                          %92 = loom.init_tensor %91[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %93 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %94 = loom.semaphore_take %93 : memref<32x?xf16> -> memref<32x?xf16>
                          %95 = loom.init_tensor %94[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %96 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %97 = loom.semaphore_take %96 : memref<?x?xf16> -> memref<?x?xf16>
                          %98 = loom.init_tensor %97[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %99 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %100 = loom.semaphore_take %99 : memref<?x?xf16> -> memref<?x?xf16>
                          %101 = scf.for %arg18 = %c0 to %89 step %c1 iter_args(%arg19 = %86) -> (tensor<?x?xf16>) {
                            %116 = arith.muli %arg18, %22 : index
                            %117 = arith.addi %116, %22 : index
                            %118 = arith.cmpi ult, %117, %88 : index
                            %119 = arith.select %118, %117, %88 : index
                            %120 = arith.subi %119, %116 : index
                            %121 = loom.alloc [%20, %120] on @L1 : memref<?x?xf16>
                            %122 = loom.semaphore_take %121 : memref<?x?xf16> -> memref<?x?xf16>
                            %123 = loom.subview %arg0[%41, %43, %64, %44, %116] [1, 1, 1, %20, %120] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x8x512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                            loom.copy %123, %122 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                            %124 = loom.bufferize_to_tensor %122[%20, %120] : memref<?x?xf16> -> tensor<?x?xf16>
                            %125 = loom.alloc [%120] on @L1 : memref<?xf16>
                            %126 = loom.semaphore_take %125 : memref<?xf16> -> memref<?xf16>
                            %127 = loom.semaphore_take %125 : memref<?xf16> -> memref<?xf16>
                            %128 = loom.subview %arg1[%41, %42, %43, %116] [1, 1, 1, %120] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %128, %127 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %129 = loom.bufferize_to_tensor %127[%120] : memref<?xf16> -> tensor<?xf16>
                            %130 = loom.broadcast ins(%55 : tensor<?xf16>) outs(%58 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %131 = loom.broadcast ins(%129 : tensor<?xf16>) outs(%92 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %127 : memref<?xf16>
                            %132 = loom.subview %arg2[%41, %42, %43, %116] [1, 1, 1, %120] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %132, %126 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %133 = loom.bufferize_to_tensor %126[%120] : memref<?xf16> -> tensor<?xf16>
                            %134 = loom.broadcast ins(%133 : tensor<?xf16>) outs(%95 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %126 : memref<?xf16>
                            %135 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%124, %130, %131, %134 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%98 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %140 = arith.subf %in_0, %in_1 : f16
                              %141 = math.exp %140 : f16
                              %142 = arith.mulf %in, %141 : f16
                              %143 = arith.mulf %142, %in_2 : f16
                              linalg.yield %143 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %94 : memref<32x?xf16>
                            loom.semaphore_give %91 : memref<32x?xf16>
                            loom.semaphore_give %122 : memref<?x?xf16>
                            loom.semaphore_give %57 : memref<?x32xf16>
                            %136 = arith.addi %116, %62 : index
                            %137 = loom.subview %arg3[%41, %136, %42, %69] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                            loom.copy %137, %100 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?x?xf16, strided<[16384, 1], offset: ?>> to memref<?x?xf16>
                            %138 = loom.bufferize_to_tensor %100[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %139 = linalg.matmul ins(%135, %138 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %100 : memref<?x?xf16>
                            loom.semaphore_give %97 : memref<?x?xf16>
                            scf.yield %139 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %102 = loom.alloc [1] on @L1 : memref<f16>
                          %103 = loom.semaphore_take %102 : memref<f16> -> memref<f16>
                          %104 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<128xf16> to memref<f16, strided<[], offset: ?>>
                          %105 = arith.addi %51, %c3 : index
                          loom.copy %104, %103 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %51], LR : [%c7, %105]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %106 = loom.bufferize_to_tensor %103[] : memref<f16> -> tensor<f16>
                          %107 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %108 = loom.semaphore_take %107 : memref<?x?xf16> -> memref<?x?xf16>
                          %109 = loom.init_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %110 = loom.subview %arg3[%41, %63, %42, %69] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          loom.copy %110, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?x?xf16, strided<[16384, 1], offset: ?>> to memref<?x?xf16>
                          %111 = loom.bufferize_to_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101, %111, %106 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%109 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %116 = arith.mulf %in_0, %in_1 : f16
                            %117 = arith.addf %in, %116 : f16
                            linalg.yield %117 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %103 : memref<f16>
                          loom.semaphore_give %82 : memref<?x?xf16>
                          %113 = loom.sync ins(%112 : tensor<?x?xf16>) outs(%79 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %108 : memref<?x?xf16>
                          %114 = loom.subview %arg7[%41, %63, %42, %69] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          %115 = loom.bufferize_to_memref %113 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %115, %114 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?x?xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          loom.semaphore_give %78 : memref<?x?xf16>
                        } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
                  } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 128 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 128 : index}} {
    func.func @_mamba_chunk_scan__x2x4_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc8_dim_y_level1_bc4_n_n(%arg0: memref<2x8x8x512x512xf16>, %arg1: memref<2x128x8x512xf16>, %arg2: memref<2x128x8x512xf16>, %arg3: memref<2x4096x128x128xf16>, %arg4: memref<2x4096x8x128xf16>, %arg5: memref<2x8x128x128x128xf16>, %arg6: memref<128xf16>, %arg7: memref<2x4096x128x128xf16>) {
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c512 = arith.constant 512 : index
      %c128 = arith.constant 128 : index
      %20 = loom.sym @tile_m {upper_bound = 512 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 128 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 128 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c128, %23 : index
      %27 = arith.ceildivui %c512, %20 : index
      %28 = arith.ceildivui %c128, %21 : index
      %29 = arith.ceildivui %c2, %24 : index
      %30 = arith.ceildivui %c8, %25 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (4) {
                %31 = arith.ceildivui %26, %c2 : index
                scf.for %arg13 = %c0 to %31 step %c1 {
                  %32 = arith.ceildivui %27, %c2 : index
                  scf.for %arg14 = %c0 to %32 step %c1 {
                    %33 = arith.ceildivui %28, %c2 : index
                    scf.for %arg15 = %c0 to %33 step %c1 {
                      %34 = arith.ceildivui %29, %c2 : index
                      scf.for %arg16 = %c0 to %34 step %c1 {
                        %35 = arith.ceildivui %30, %c4 : index
                        scf.for %arg17 = %c0 to %35 step %c1 {
                          %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                          %37 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                          %38 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg10, %arg15)
                          %39 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg11, %arg16)
                          %40 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg12, %arg17)
                          %41 = arith.muli %39, %24 : index
                          %42 = arith.muli %36, %23 : index
                          %43 = arith.muli %40, %25 : index
                          %44 = arith.muli %37, %20 : index
                          %45 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %46 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg12, %c2 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg11, %c2 : index
                          %51 = arith.muli %arg8, %c4 : index
                          %52 = arith.addi %50, %51 : index
                          %53 = arith.addi %50, %c1 : index
                          %54 = arith.addi %53, %51 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %55 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %56 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %57 = loom.semaphore_take %56 : memref<?x32xf16> -> memref<?x32xf16>
                          %58 = loom.init_tensor %57[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %59 = loom.semaphore_take %56 : memref<?x32xf16> -> memref<?x32xf16>
                          %60 = loom.init_tensor %59[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %61 = loom.broadcast ins(%55 : tensor<?xf16>) outs(%60 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %62 = arith.muli %43, %c512 : index
                          %63 = arith.addi %44, %62 : index
                          %64 = arith.divui %42, %c16 : index
                          %65 = loom.alloc [%20, 128] on @L1 : memref<?x128xf16>
                          %66 = loom.semaphore_take %65 : memref<?x128xf16> -> memref<?x128xf16>
                          %67 = loom.subview %arg4[%41, %63, %64, 0] [1, %20, 1, 128] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x4096x8x128xf16> to memref<?x128xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %67, %66 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?x128xf16, strided<[1024, 1], offset: ?>> to memref<?x128xf16>
                          %68 = loom.bufferize_to_tensor %66[%20, 128] : memref<?x128xf16> -> tensor<?x128xf16>
                          %69 = arith.muli %38, %21 : index
                          %70 = loom.alloc [128, %21] on @L1 : memref<128x?xf16>
                          %71 = loom.semaphore_take %70 : memref<128x?xf16> -> memref<128x?xf16>
                          %72 = loom.subview %arg5[%41, %43, %42, 0, %69] [1, 1, 1, 128, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x128x128x128xf16> to memref<128x?xf16, strided<[128, 1], offset: ?>>
                          %73 = arith.addi %48, %c1 : index
                          %74 = arith.addi %arg10, %50 : index
                          %75 = arith.addi %74, %51 : index
                          loom.copy %72, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%48, %75], LR : [%73, %75]) : memref<128x?xf16, strided<[128, 1], offset: ?>> to memref<128x?xf16>
                          %76 = loom.bufferize_to_tensor %71[128, %21] : memref<128x?xf16> -> tensor<128x?xf16>
                          %77 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %79 = loom.init_tensor %78[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %80 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %81 = loom.init_tensor %80[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %82 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %83 = loom.init_tensor %82[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %84 = linalg.fill ins(%cst : f16) outs(%81 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %85 = linalg.matmul ins(%68, %76 : tensor<?x128xf16>, tensor<128x?xf16>) outs(%84 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %71 : memref<128x?xf16>
                          loom.semaphore_give %66 : memref<?x128xf16>
                          %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%85, %61 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%83 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %116 = math.exp %in_0 : f16
                            %117 = arith.mulf %in, %116 : f16
                            linalg.yield %117 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %80 : memref<?x?xf16>
                          loom.semaphore_give %59 : memref<?x32xf16>
                          %87 = arith.addi %37, %c1 : index
                          %88 = arith.muli %87, %20 : index
                          %89 = arith.ceildivui %88, %22 : index
                          %90 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %91 = loom.semaphore_take %90 : memref<32x?xf16> -> memref<32x?xf16>
                          %92 = loom.init_tensor %91[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %93 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %94 = loom.semaphore_take %93 : memref<32x?xf16> -> memref<32x?xf16>
                          %95 = loom.init_tensor %94[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %96 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %97 = loom.semaphore_take %96 : memref<?x?xf16> -> memref<?x?xf16>
                          %98 = loom.init_tensor %97[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %99 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %100 = loom.semaphore_take %99 : memref<?x?xf16> -> memref<?x?xf16>
                          %101 = scf.for %arg18 = %c0 to %89 step %c1 iter_args(%arg19 = %86) -> (tensor<?x?xf16>) {
                            %116 = arith.muli %arg18, %22 : index
                            %117 = arith.addi %116, %22 : index
                            %118 = arith.cmpi ult, %117, %88 : index
                            %119 = arith.select %118, %117, %88 : index
                            %120 = arith.subi %119, %116 : index
                            %121 = loom.alloc [%20, %120] on @L1 : memref<?x?xf16>
                            %122 = loom.semaphore_take %121 : memref<?x?xf16> -> memref<?x?xf16>
                            %123 = loom.subview %arg0[%41, %43, %64, %44, %116] [1, 1, 1, %20, %120] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x8x512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                            loom.copy %123, %122 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                            %124 = loom.bufferize_to_tensor %122[%20, %120] : memref<?x?xf16> -> tensor<?x?xf16>
                            %125 = loom.alloc [%120] on @L1 : memref<?xf16>
                            %126 = loom.semaphore_take %125 : memref<?xf16> -> memref<?xf16>
                            %127 = loom.semaphore_take %125 : memref<?xf16> -> memref<?xf16>
                            %128 = loom.subview %arg1[%41, %42, %43, %116] [1, 1, 1, %120] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %128, %127 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %129 = loom.bufferize_to_tensor %127[%120] : memref<?xf16> -> tensor<?xf16>
                            %130 = loom.broadcast ins(%55 : tensor<?xf16>) outs(%58 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %131 = loom.broadcast ins(%129 : tensor<?xf16>) outs(%92 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %127 : memref<?xf16>
                            %132 = loom.subview %arg2[%41, %42, %43, %116] [1, 1, 1, %120] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %132, %126 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %133 = loom.bufferize_to_tensor %126[%120] : memref<?xf16> -> tensor<?xf16>
                            %134 = loom.broadcast ins(%133 : tensor<?xf16>) outs(%95 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %126 : memref<?xf16>
                            %135 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%124, %130, %131, %134 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%98 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %140 = arith.subf %in_0, %in_1 : f16
                              %141 = math.exp %140 : f16
                              %142 = arith.mulf %in, %141 : f16
                              %143 = arith.mulf %142, %in_2 : f16
                              linalg.yield %143 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %94 : memref<32x?xf16>
                            loom.semaphore_give %91 : memref<32x?xf16>
                            loom.semaphore_give %122 : memref<?x?xf16>
                            loom.semaphore_give %57 : memref<?x32xf16>
                            %136 = arith.addi %116, %62 : index
                            %137 = loom.subview %arg3[%41, %136, %42, %69] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                            loom.copy %137, %100 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?x?xf16, strided<[16384, 1], offset: ?>> to memref<?x?xf16>
                            %138 = loom.bufferize_to_tensor %100[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %139 = linalg.matmul ins(%135, %138 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %100 : memref<?x?xf16>
                            loom.semaphore_give %97 : memref<?x?xf16>
                            scf.yield %139 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %102 = loom.alloc [1] on @L1 : memref<f16>
                          %103 = loom.semaphore_take %102 : memref<f16> -> memref<f16>
                          %104 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<128xf16> to memref<f16, strided<[], offset: ?>>
                          %105 = arith.addi %51, %c3 : index
                          loom.copy %104, %103 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %51], LR : [%c7, %105]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %106 = loom.bufferize_to_tensor %103[] : memref<f16> -> tensor<f16>
                          %107 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %108 = loom.semaphore_take %107 : memref<?x?xf16> -> memref<?x?xf16>
                          %109 = loom.init_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %110 = loom.subview %arg3[%41, %63, %42, %69] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          loom.copy %110, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?x?xf16, strided<[16384, 1], offset: ?>> to memref<?x?xf16>
                          %111 = loom.bufferize_to_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101, %111, %106 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%109 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %116 = arith.mulf %in_0, %in_1 : f16
                            %117 = arith.addf %in, %116 : f16
                            linalg.yield %117 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %103 : memref<f16>
                          loom.semaphore_give %82 : memref<?x?xf16>
                          %113 = loom.sync ins(%112 : tensor<?x?xf16>) outs(%79 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %108 : memref<?x?xf16>
                          %114 = loom.subview %arg7[%41, %63, %42, %69] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          %115 = loom.bufferize_to_memref %113 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %115, %114 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?x?xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          loom.semaphore_give %78 : memref<?x?xf16>
                        } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
                  } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 128 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 128 : index}} {
    func.func @_mamba_chunk_scan__x4x2_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc4_n_n_n_n_dim_x_level1_bc8_dim_y_level1_bc4_n_n(%arg0: memref<2x8x8x512x512xf16>, %arg1: memref<2x128x8x512xf16>, %arg2: memref<2x128x8x512xf16>, %arg3: memref<2x4096x128x128xf16>, %arg4: memref<2x4096x8x128xf16>, %arg5: memref<2x8x128x128x128xf16>, %arg6: memref<128xf16>, %arg7: memref<2x4096x128x128xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c512 = arith.constant 512 : index
      %c128 = arith.constant 128 : index
      %20 = loom.sym @tile_m {upper_bound = 512 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 128 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 128 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c128, %23 : index
      %27 = arith.ceildivui %c512, %20 : index
      %28 = arith.ceildivui %c128, %21 : index
      %29 = arith.ceildivui %c2, %24 : index
      %30 = arith.ceildivui %c8, %25 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (4) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                %31 = arith.ceildivui %26, %c2 : index
                scf.for %arg13 = %c0 to %31 step %c1 {
                  %32 = arith.ceildivui %27, %c4 : index
                  scf.for %arg14 = %c0 to %32 step %c1 {
                    %33 = arith.ceildivui %28, %c2 : index
                    scf.for %arg15 = %c0 to %33 step %c1 {
                      %34 = arith.ceildivui %29, %c2 : index
                      scf.for %arg16 = %c0 to %34 step %c1 {
                        %35 = arith.ceildivui %30, %c2 : index
                        scf.for %arg17 = %c0 to %35 step %c1 {
                          %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                          %37 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg9, %arg14)
                          %38 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg10, %arg15)
                          %39 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg11, %arg16)
                          %40 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg12, %arg17)
                          %41 = arith.muli %39, %24 : index
                          %42 = arith.muli %36, %23 : index
                          %43 = arith.muli %40, %25 : index
                          %44 = arith.muli %37, %20 : index
                          %45 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %46 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg11, %c4 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg12, %c2 : index
                          %51 = arith.muli %arg8, %c4 : index
                          %52 = arith.addi %50, %51 : index
                          %53 = arith.addi %50, %c1 : index
                          %54 = arith.addi %53, %51 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %55 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %56 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %57 = loom.semaphore_take %56 : memref<?x32xf16> -> memref<?x32xf16>
                          %58 = loom.init_tensor %57[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %59 = loom.semaphore_take %56 : memref<?x32xf16> -> memref<?x32xf16>
                          %60 = loom.init_tensor %59[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %61 = loom.broadcast ins(%55 : tensor<?xf16>) outs(%60 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %62 = arith.muli %43, %c512 : index
                          %63 = arith.addi %44, %62 : index
                          %64 = arith.divui %42, %c16 : index
                          %65 = loom.alloc [%20, 128] on @L1 : memref<?x128xf16>
                          %66 = loom.semaphore_take %65 : memref<?x128xf16> -> memref<?x128xf16>
                          %67 = loom.subview %arg4[%41, %63, %64, 0] [1, %20, 1, 128] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x4096x8x128xf16> to memref<?x128xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %67, %66 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?x128xf16, strided<[1024, 1], offset: ?>> to memref<?x128xf16>
                          %68 = loom.bufferize_to_tensor %66[%20, 128] : memref<?x128xf16> -> tensor<?x128xf16>
                          %69 = arith.muli %38, %21 : index
                          %70 = loom.alloc [128, %21] on @L1 : memref<128x?xf16>
                          %71 = loom.semaphore_take %70 : memref<128x?xf16> -> memref<128x?xf16>
                          %72 = loom.subview %arg5[%41, %43, %42, 0, %69] [1, 1, 1, 128, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x128x128x128xf16> to memref<128x?xf16, strided<[128, 1], offset: ?>>
                          %73 = arith.addi %48, %c3 : index
                          %74 = arith.addi %arg10, %50 : index
                          %75 = arith.addi %74, %51 : index
                          loom.copy %72, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%48, %75], LR : [%73, %75]) : memref<128x?xf16, strided<[128, 1], offset: ?>> to memref<128x?xf16>
                          %76 = loom.bufferize_to_tensor %71[128, %21] : memref<128x?xf16> -> tensor<128x?xf16>
                          %77 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %79 = loom.init_tensor %78[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %80 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %81 = loom.init_tensor %80[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %82 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %83 = loom.init_tensor %82[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %84 = linalg.fill ins(%cst : f16) outs(%81 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %85 = linalg.matmul ins(%68, %76 : tensor<?x128xf16>, tensor<128x?xf16>) outs(%84 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %71 : memref<128x?xf16>
                          loom.semaphore_give %66 : memref<?x128xf16>
                          %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%85, %61 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%83 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %116 = math.exp %in_0 : f16
                            %117 = arith.mulf %in, %116 : f16
                            linalg.yield %117 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %80 : memref<?x?xf16>
                          loom.semaphore_give %59 : memref<?x32xf16>
                          %87 = arith.addi %37, %c1 : index
                          %88 = arith.muli %87, %20 : index
                          %89 = arith.ceildivui %88, %22 : index
                          %90 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %91 = loom.semaphore_take %90 : memref<32x?xf16> -> memref<32x?xf16>
                          %92 = loom.init_tensor %91[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %93 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %94 = loom.semaphore_take %93 : memref<32x?xf16> -> memref<32x?xf16>
                          %95 = loom.init_tensor %94[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %96 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %97 = loom.semaphore_take %96 : memref<?x?xf16> -> memref<?x?xf16>
                          %98 = loom.init_tensor %97[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %99 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %100 = loom.semaphore_take %99 : memref<?x?xf16> -> memref<?x?xf16>
                          %101 = scf.for %arg18 = %c0 to %89 step %c1 iter_args(%arg19 = %86) -> (tensor<?x?xf16>) {
                            %116 = arith.muli %arg18, %22 : index
                            %117 = arith.addi %116, %22 : index
                            %118 = arith.cmpi ult, %117, %88 : index
                            %119 = arith.select %118, %117, %88 : index
                            %120 = arith.subi %119, %116 : index
                            %121 = loom.alloc [%20, %120] on @L1 : memref<?x?xf16>
                            %122 = loom.semaphore_take %121 : memref<?x?xf16> -> memref<?x?xf16>
                            %123 = loom.subview %arg0[%41, %43, %64, %44, %116] [1, 1, 1, %20, %120] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x8x512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                            loom.copy %123, %122 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                            %124 = loom.bufferize_to_tensor %122[%20, %120] : memref<?x?xf16> -> tensor<?x?xf16>
                            %125 = loom.alloc [%120] on @L1 : memref<?xf16>
                            %126 = loom.semaphore_take %125 : memref<?xf16> -> memref<?xf16>
                            %127 = loom.semaphore_take %125 : memref<?xf16> -> memref<?xf16>
                            %128 = loom.subview %arg1[%41, %42, %43, %116] [1, 1, 1, %120] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %128, %127 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %129 = loom.bufferize_to_tensor %127[%120] : memref<?xf16> -> tensor<?xf16>
                            %130 = loom.broadcast ins(%55 : tensor<?xf16>) outs(%58 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %131 = loom.broadcast ins(%129 : tensor<?xf16>) outs(%92 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %127 : memref<?xf16>
                            %132 = loom.subview %arg2[%41, %42, %43, %116] [1, 1, 1, %120] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %132, %126 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %133 = loom.bufferize_to_tensor %126[%120] : memref<?xf16> -> tensor<?xf16>
                            %134 = loom.broadcast ins(%133 : tensor<?xf16>) outs(%95 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %126 : memref<?xf16>
                            %135 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%124, %130, %131, %134 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%98 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %140 = arith.subf %in_0, %in_1 : f16
                              %141 = math.exp %140 : f16
                              %142 = arith.mulf %in, %141 : f16
                              %143 = arith.mulf %142, %in_2 : f16
                              linalg.yield %143 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %94 : memref<32x?xf16>
                            loom.semaphore_give %91 : memref<32x?xf16>
                            loom.semaphore_give %122 : memref<?x?xf16>
                            loom.semaphore_give %57 : memref<?x32xf16>
                            %136 = arith.addi %116, %62 : index
                            %137 = loom.subview %arg3[%41, %136, %42, %69] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                            loom.copy %137, %100 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?x?xf16, strided<[16384, 1], offset: ?>> to memref<?x?xf16>
                            %138 = loom.bufferize_to_tensor %100[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %139 = linalg.matmul ins(%135, %138 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %100 : memref<?x?xf16>
                            loom.semaphore_give %97 : memref<?x?xf16>
                            scf.yield %139 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %102 = loom.alloc [1] on @L1 : memref<f16>
                          %103 = loom.semaphore_take %102 : memref<f16> -> memref<f16>
                          %104 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<128xf16> to memref<f16, strided<[], offset: ?>>
                          %105 = arith.addi %51, %c3 : index
                          loom.copy %104, %103 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %51], LR : [%c7, %105]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %106 = loom.bufferize_to_tensor %103[] : memref<f16> -> tensor<f16>
                          %107 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %108 = loom.semaphore_take %107 : memref<?x?xf16> -> memref<?x?xf16>
                          %109 = loom.init_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %110 = loom.subview %arg3[%41, %63, %42, %69] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          loom.copy %110, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?x?xf16, strided<[16384, 1], offset: ?>> to memref<?x?xf16>
                          %111 = loom.bufferize_to_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101, %111, %106 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%109 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %116 = arith.mulf %in_0, %in_1 : f16
                            %117 = arith.addf %in, %116 : f16
                            linalg.yield %117 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %103 : memref<f16>
                          loom.semaphore_give %82 : memref<?x?xf16>
                          %113 = loom.sync ins(%112 : tensor<?x?xf16>) outs(%79 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %108 : memref<?x?xf16>
                          %114 = loom.subview %arg7[%41, %63, %42, %69] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          %115 = loom.bufferize_to_memref %113 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %115, %114 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?x?xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          loom.semaphore_give %78 : memref<?x?xf16>
                        } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
                  } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 128 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 128 : index}} {
    func.func @_mamba_chunk_scan__x4x2_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc4_n_n_n_n_dim_x_level1_bc8_dim_y_level1_bc4_n_n(%arg0: memref<2x8x8x512x512xf16>, %arg1: memref<2x128x8x512xf16>, %arg2: memref<2x128x8x512xf16>, %arg3: memref<2x4096x128x128xf16>, %arg4: memref<2x4096x8x128xf16>, %arg5: memref<2x8x128x128x128xf16>, %arg6: memref<128xf16>, %arg7: memref<2x4096x128x128xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c512 = arith.constant 512 : index
      %c128 = arith.constant 128 : index
      %20 = loom.sym @tile_m {upper_bound = 512 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 128 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 128 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c128, %23 : index
      %27 = arith.ceildivui %c512, %20 : index
      %28 = arith.ceildivui %c128, %21 : index
      %29 = arith.ceildivui %c2, %24 : index
      %30 = arith.ceildivui %c8, %25 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (4) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                %31 = arith.ceildivui %26, %c2 : index
                scf.for %arg13 = %c0 to %31 step %c1 {
                  %32 = arith.ceildivui %27, %c4 : index
                  scf.for %arg14 = %c0 to %32 step %c1 {
                    %33 = arith.ceildivui %28, %c2 : index
                    scf.for %arg15 = %c0 to %33 step %c1 {
                      %34 = arith.ceildivui %29, %c2 : index
                      scf.for %arg16 = %c0 to %34 step %c1 {
                        %35 = arith.ceildivui %30, %c2 : index
                        scf.for %arg17 = %c0 to %35 step %c1 {
                          %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                          %37 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg9, %arg14)
                          %38 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg10, %arg15)
                          %39 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg11, %arg16)
                          %40 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg12, %arg17)
                          %41 = arith.muli %39, %24 : index
                          %42 = arith.muli %36, %23 : index
                          %43 = arith.muli %40, %25 : index
                          %44 = arith.muli %37, %20 : index
                          %45 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %46 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg12, %c4 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg11, %c2 : index
                          %51 = arith.muli %arg8, %c4 : index
                          %52 = arith.addi %50, %51 : index
                          %53 = arith.addi %50, %c1 : index
                          %54 = arith.addi %53, %51 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %55 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %56 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %57 = loom.semaphore_take %56 : memref<?x32xf16> -> memref<?x32xf16>
                          %58 = loom.init_tensor %57[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %59 = loom.semaphore_take %56 : memref<?x32xf16> -> memref<?x32xf16>
                          %60 = loom.init_tensor %59[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %61 = loom.broadcast ins(%55 : tensor<?xf16>) outs(%60 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %62 = arith.muli %43, %c512 : index
                          %63 = arith.addi %44, %62 : index
                          %64 = arith.divui %42, %c16 : index
                          %65 = loom.alloc [%20, 128] on @L1 : memref<?x128xf16>
                          %66 = loom.semaphore_take %65 : memref<?x128xf16> -> memref<?x128xf16>
                          %67 = loom.subview %arg4[%41, %63, %64, 0] [1, %20, 1, 128] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x4096x8x128xf16> to memref<?x128xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %67, %66 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?x128xf16, strided<[1024, 1], offset: ?>> to memref<?x128xf16>
                          %68 = loom.bufferize_to_tensor %66[%20, 128] : memref<?x128xf16> -> tensor<?x128xf16>
                          %69 = arith.muli %38, %21 : index
                          %70 = loom.alloc [128, %21] on @L1 : memref<128x?xf16>
                          %71 = loom.semaphore_take %70 : memref<128x?xf16> -> memref<128x?xf16>
                          %72 = loom.subview %arg5[%41, %43, %42, 0, %69] [1, 1, 1, 128, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x128x128x128xf16> to memref<128x?xf16, strided<[128, 1], offset: ?>>
                          %73 = arith.addi %48, %c3 : index
                          %74 = arith.addi %arg10, %50 : index
                          %75 = arith.addi %74, %51 : index
                          loom.copy %72, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%48, %75], LR : [%73, %75]) : memref<128x?xf16, strided<[128, 1], offset: ?>> to memref<128x?xf16>
                          %76 = loom.bufferize_to_tensor %71[128, %21] : memref<128x?xf16> -> tensor<128x?xf16>
                          %77 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %79 = loom.init_tensor %78[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %80 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %81 = loom.init_tensor %80[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %82 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %83 = loom.init_tensor %82[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %84 = linalg.fill ins(%cst : f16) outs(%81 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %85 = linalg.matmul ins(%68, %76 : tensor<?x128xf16>, tensor<128x?xf16>) outs(%84 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %71 : memref<128x?xf16>
                          loom.semaphore_give %66 : memref<?x128xf16>
                          %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%85, %61 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%83 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %116 = math.exp %in_0 : f16
                            %117 = arith.mulf %in, %116 : f16
                            linalg.yield %117 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %80 : memref<?x?xf16>
                          loom.semaphore_give %59 : memref<?x32xf16>
                          %87 = arith.addi %37, %c1 : index
                          %88 = arith.muli %87, %20 : index
                          %89 = arith.ceildivui %88, %22 : index
                          %90 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %91 = loom.semaphore_take %90 : memref<32x?xf16> -> memref<32x?xf16>
                          %92 = loom.init_tensor %91[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %93 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %94 = loom.semaphore_take %93 : memref<32x?xf16> -> memref<32x?xf16>
                          %95 = loom.init_tensor %94[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %96 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %97 = loom.semaphore_take %96 : memref<?x?xf16> -> memref<?x?xf16>
                          %98 = loom.init_tensor %97[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %99 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %100 = loom.semaphore_take %99 : memref<?x?xf16> -> memref<?x?xf16>
                          %101 = scf.for %arg18 = %c0 to %89 step %c1 iter_args(%arg19 = %86) -> (tensor<?x?xf16>) {
                            %116 = arith.muli %arg18, %22 : index
                            %117 = arith.addi %116, %22 : index
                            %118 = arith.cmpi ult, %117, %88 : index
                            %119 = arith.select %118, %117, %88 : index
                            %120 = arith.subi %119, %116 : index
                            %121 = loom.alloc [%20, %120] on @L1 : memref<?x?xf16>
                            %122 = loom.semaphore_take %121 : memref<?x?xf16> -> memref<?x?xf16>
                            %123 = loom.subview %arg0[%41, %43, %64, %44, %116] [1, 1, 1, %20, %120] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x8x512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                            loom.copy %123, %122 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                            %124 = loom.bufferize_to_tensor %122[%20, %120] : memref<?x?xf16> -> tensor<?x?xf16>
                            %125 = loom.alloc [%120] on @L1 : memref<?xf16>
                            %126 = loom.semaphore_take %125 : memref<?xf16> -> memref<?xf16>
                            %127 = loom.semaphore_take %125 : memref<?xf16> -> memref<?xf16>
                            %128 = loom.subview %arg1[%41, %42, %43, %116] [1, 1, 1, %120] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %128, %127 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %129 = loom.bufferize_to_tensor %127[%120] : memref<?xf16> -> tensor<?xf16>
                            %130 = loom.broadcast ins(%55 : tensor<?xf16>) outs(%58 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %131 = loom.broadcast ins(%129 : tensor<?xf16>) outs(%92 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %127 : memref<?xf16>
                            %132 = loom.subview %arg2[%41, %42, %43, %116] [1, 1, 1, %120] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %132, %126 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %133 = loom.bufferize_to_tensor %126[%120] : memref<?xf16> -> tensor<?xf16>
                            %134 = loom.broadcast ins(%133 : tensor<?xf16>) outs(%95 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %126 : memref<?xf16>
                            %135 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%124, %130, %131, %134 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%98 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %140 = arith.subf %in_0, %in_1 : f16
                              %141 = math.exp %140 : f16
                              %142 = arith.mulf %in, %141 : f16
                              %143 = arith.mulf %142, %in_2 : f16
                              linalg.yield %143 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %94 : memref<32x?xf16>
                            loom.semaphore_give %91 : memref<32x?xf16>
                            loom.semaphore_give %122 : memref<?x?xf16>
                            loom.semaphore_give %57 : memref<?x32xf16>
                            %136 = arith.addi %116, %62 : index
                            %137 = loom.subview %arg3[%41, %136, %42, %69] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                            loom.copy %137, %100 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?x?xf16, strided<[16384, 1], offset: ?>> to memref<?x?xf16>
                            %138 = loom.bufferize_to_tensor %100[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %139 = linalg.matmul ins(%135, %138 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %100 : memref<?x?xf16>
                            loom.semaphore_give %97 : memref<?x?xf16>
                            scf.yield %139 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %102 = loom.alloc [1] on @L1 : memref<f16>
                          %103 = loom.semaphore_take %102 : memref<f16> -> memref<f16>
                          %104 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<128xf16> to memref<f16, strided<[], offset: ?>>
                          %105 = arith.addi %51, %c3 : index
                          loom.copy %104, %103 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %51], LR : [%c7, %105]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %106 = loom.bufferize_to_tensor %103[] : memref<f16> -> tensor<f16>
                          %107 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %108 = loom.semaphore_take %107 : memref<?x?xf16> -> memref<?x?xf16>
                          %109 = loom.init_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %110 = loom.subview %arg3[%41, %63, %42, %69] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          loom.copy %110, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?x?xf16, strided<[16384, 1], offset: ?>> to memref<?x?xf16>
                          %111 = loom.bufferize_to_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101, %111, %106 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%109 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %116 = arith.mulf %in_0, %in_1 : f16
                            %117 = arith.addf %in, %116 : f16
                            linalg.yield %117 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %103 : memref<f16>
                          loom.semaphore_give %82 : memref<?x?xf16>
                          %113 = loom.sync ins(%112 : tensor<?x?xf16>) outs(%79 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %108 : memref<?x?xf16>
                          %114 = loom.subview %arg7[%41, %63, %42, %69] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          %115 = loom.bufferize_to_memref %113 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %115, %114 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%49, %75], LR : [%49, %75]) : memref<?x?xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          loom.semaphore_give %78 : memref<?x?xf16>
                        } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
                  } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 128 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 128 : index}} {
    func.func @_mamba_chunk_scan__x2x2x2_y2y4__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc4_dim_y_level1_bc8_n_n(%arg0: memref<2x8x8x512x512xf16>, %arg1: memref<2x128x8x512xf16>, %arg2: memref<2x128x8x512xf16>, %arg3: memref<2x4096x128x128xf16>, %arg4: memref<2x4096x8x128xf16>, %arg5: memref<2x8x128x128x128xf16>, %arg6: memref<128xf16>, %arg7: memref<2x4096x128x128xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c512 = arith.constant 512 : index
      %c128 = arith.constant 128 : index
      %20 = loom.sym @tile_m {upper_bound = 512 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 128 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 128 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c128, %23 : index
      %27 = arith.ceildivui %c512, %20 : index
      %28 = arith.ceildivui %c128, %21 : index
      %29 = arith.ceildivui %c2, %24 : index
      %30 = arith.ceildivui %c8, %25 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (4) {
                %31 = arith.ceildivui %26, %c2 : index
                scf.for %arg13 = %c0 to %31 step %c1 {
                  %32 = arith.ceildivui %27, %c2 : index
                  scf.for %arg14 = %c0 to %32 step %c1 {
                    %33 = arith.ceildivui %28, %c2 : index
                    scf.for %arg15 = %c0 to %33 step %c1 {
                      %34 = arith.ceildivui %29, %c2 : index
                      scf.for %arg16 = %c0 to %34 step %c1 {
                        %35 = arith.ceildivui %30, %c4 : index
                        scf.for %arg17 = %c0 to %35 step %c1 {
                          %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                          %37 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                          %38 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg10, %arg15)
                          %39 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg11, %arg16)
                          %40 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg12, %arg17)
                          %41 = arith.muli %39, %24 : index
                          %42 = arith.muli %36, %23 : index
                          %43 = arith.muli %40, %25 : index
                          %44 = arith.muli %37, %20 : index
                          %45 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %46 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg11, %c2 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg8, %c4 : index
                          %51 = arith.addi %49, %50 : index
                          %52 = arith.muli %arg12, %c2 : index
                          %53 = arith.addi %52, %c1 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %54 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %55 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %56 = loom.semaphore_take %55 : memref<?x32xf16> -> memref<?x32xf16>
                          %57 = loom.init_tensor %56[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %58 = loom.semaphore_take %55 : memref<?x32xf16> -> memref<?x32xf16>
                          %59 = loom.init_tensor %58[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %60 = loom.broadcast ins(%54 : tensor<?xf16>) outs(%59 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %61 = arith.muli %43, %c512 : index
                          %62 = arith.addi %44, %61 : index
                          %63 = arith.divui %42, %c16 : index
                          %64 = loom.alloc [%20, 128] on @L1 : memref<?x128xf16>
                          %65 = loom.semaphore_take %64 : memref<?x128xf16> -> memref<?x128xf16>
                          %66 = loom.subview %arg4[%41, %62, %63, 0] [1, %20, 1, 128] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x4096x8x128xf16> to memref<?x128xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %66, %65 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?x128xf16, strided<[1024, 1], offset: ?>> to memref<?x128xf16>
                          %67 = loom.bufferize_to_tensor %65[%20, 128] : memref<?x128xf16> -> tensor<?x128xf16>
                          %68 = arith.muli %38, %21 : index
                          %69 = loom.alloc [128, %21] on @L1 : memref<128x?xf16>
                          %70 = loom.semaphore_take %69 : memref<128x?xf16> -> memref<128x?xf16>
                          %71 = loom.subview %arg5[%41, %43, %42, 0, %68] [1, 1, 1, 128, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x128x128x128xf16> to memref<128x?xf16, strided<[128, 1], offset: ?>>
                          %72 = arith.addi %48, %50 : index
                          %73 = arith.addi %48, %c1 : index
                          %74 = arith.addi %73, %50 : index
                          %75 = arith.addi %arg10, %52 : index
                          loom.copy %71, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%72, %75], LR : [%74, %75]) : memref<128x?xf16, strided<[128, 1], offset: ?>> to memref<128x?xf16>
                          %76 = loom.bufferize_to_tensor %70[128, %21] : memref<128x?xf16> -> tensor<128x?xf16>
                          %77 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %79 = loom.init_tensor %78[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %80 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %81 = loom.init_tensor %80[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %82 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %83 = loom.init_tensor %82[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %84 = linalg.fill ins(%cst : f16) outs(%81 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %85 = linalg.matmul ins(%67, %76 : tensor<?x128xf16>, tensor<128x?xf16>) outs(%84 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %70 : memref<128x?xf16>
                          loom.semaphore_give %65 : memref<?x128xf16>
                          %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%85, %60 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%83 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %116 = math.exp %in_0 : f16
                            %117 = arith.mulf %in, %116 : f16
                            linalg.yield %117 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %80 : memref<?x?xf16>
                          loom.semaphore_give %58 : memref<?x32xf16>
                          %87 = arith.addi %37, %c1 : index
                          %88 = arith.muli %87, %20 : index
                          %89 = arith.ceildivui %88, %22 : index
                          %90 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %91 = loom.semaphore_take %90 : memref<32x?xf16> -> memref<32x?xf16>
                          %92 = loom.init_tensor %91[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %93 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %94 = loom.semaphore_take %93 : memref<32x?xf16> -> memref<32x?xf16>
                          %95 = loom.init_tensor %94[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %96 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %97 = loom.semaphore_take %96 : memref<?x?xf16> -> memref<?x?xf16>
                          %98 = loom.init_tensor %97[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %99 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %100 = loom.semaphore_take %99 : memref<?x?xf16> -> memref<?x?xf16>
                          %101 = scf.for %arg18 = %c0 to %89 step %c1 iter_args(%arg19 = %86) -> (tensor<?x?xf16>) {
                            %116 = arith.muli %arg18, %22 : index
                            %117 = arith.addi %116, %22 : index
                            %118 = arith.cmpi ult, %117, %88 : index
                            %119 = arith.select %118, %117, %88 : index
                            %120 = arith.subi %119, %116 : index
                            %121 = loom.alloc [%20, %120] on @L1 : memref<?x?xf16>
                            %122 = loom.semaphore_take %121 : memref<?x?xf16> -> memref<?x?xf16>
                            %123 = loom.subview %arg0[%41, %43, %63, %44, %116] [1, 1, 1, %20, %120] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x8x512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                            loom.copy %123, %122 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                            %124 = loom.bufferize_to_tensor %122[%20, %120] : memref<?x?xf16> -> tensor<?x?xf16>
                            %125 = loom.alloc [%120] on @L1 : memref<?xf16>
                            %126 = loom.semaphore_take %125 : memref<?xf16> -> memref<?xf16>
                            %127 = loom.semaphore_take %125 : memref<?xf16> -> memref<?xf16>
                            %128 = loom.subview %arg1[%41, %42, %43, %116] [1, 1, 1, %120] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %128, %127 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %129 = loom.bufferize_to_tensor %127[%120] : memref<?xf16> -> tensor<?xf16>
                            %130 = loom.broadcast ins(%54 : tensor<?xf16>) outs(%57 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %131 = loom.broadcast ins(%129 : tensor<?xf16>) outs(%92 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %127 : memref<?xf16>
                            %132 = loom.subview %arg2[%41, %42, %43, %116] [1, 1, 1, %120] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %132, %126 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %133 = loom.bufferize_to_tensor %126[%120] : memref<?xf16> -> tensor<?xf16>
                            %134 = loom.broadcast ins(%133 : tensor<?xf16>) outs(%95 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %126 : memref<?xf16>
                            %135 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%124, %130, %131, %134 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%98 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %140 = arith.subf %in_0, %in_1 : f16
                              %141 = math.exp %140 : f16
                              %142 = arith.mulf %in, %141 : f16
                              %143 = arith.mulf %142, %in_2 : f16
                              linalg.yield %143 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %94 : memref<32x?xf16>
                            loom.semaphore_give %91 : memref<32x?xf16>
                            loom.semaphore_give %122 : memref<?x?xf16>
                            loom.semaphore_give %56 : memref<?x32xf16>
                            %136 = arith.addi %116, %61 : index
                            %137 = loom.subview %arg3[%41, %136, %42, %68] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                            loom.copy %137, %100 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?x?xf16, strided<[16384, 1], offset: ?>> to memref<?x?xf16>
                            %138 = loom.bufferize_to_tensor %100[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %139 = linalg.matmul ins(%135, %138 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %100 : memref<?x?xf16>
                            loom.semaphore_give %97 : memref<?x?xf16>
                            scf.yield %139 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %102 = loom.alloc [1] on @L1 : memref<f16>
                          %103 = loom.semaphore_take %102 : memref<f16> -> memref<f16>
                          %104 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<128xf16> to memref<f16, strided<[], offset: ?>>
                          %105 = arith.addi %50, %c3 : index
                          loom.copy %104, %103 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%50, %c0], LR : [%105, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %106 = loom.bufferize_to_tensor %103[] : memref<f16> -> tensor<f16>
                          %107 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %108 = loom.semaphore_take %107 : memref<?x?xf16> -> memref<?x?xf16>
                          %109 = loom.init_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %110 = loom.subview %arg3[%41, %62, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          loom.copy %110, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?x?xf16, strided<[16384, 1], offset: ?>> to memref<?x?xf16>
                          %111 = loom.bufferize_to_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101, %111, %106 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%109 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %116 = arith.mulf %in_0, %in_1 : f16
                            %117 = arith.addf %in, %116 : f16
                            linalg.yield %117 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %103 : memref<f16>
                          loom.semaphore_give %82 : memref<?x?xf16>
                          %113 = loom.sync ins(%112 : tensor<?x?xf16>) outs(%79 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %108 : memref<?x?xf16>
                          %114 = loom.subview %arg7[%41, %62, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          %115 = loom.bufferize_to_memref %113 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %115, %114 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?x?xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          loom.semaphore_give %78 : memref<?x?xf16>
                        } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
                  } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 128 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 128 : index}} {
    func.func @_mamba_chunk_scan__x2x2x2_y2y4__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc4_dim_y_level1_bc8_n_n(%arg0: memref<2x8x8x512x512xf16>, %arg1: memref<2x128x8x512xf16>, %arg2: memref<2x128x8x512xf16>, %arg3: memref<2x4096x128x128xf16>, %arg4: memref<2x4096x8x128xf16>, %arg5: memref<2x8x128x128x128xf16>, %arg6: memref<128xf16>, %arg7: memref<2x4096x128x128xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c512 = arith.constant 512 : index
      %c128 = arith.constant 128 : index
      %20 = loom.sym @tile_m {upper_bound = 512 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 128 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 128 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c128, %23 : index
      %27 = arith.ceildivui %c512, %20 : index
      %28 = arith.ceildivui %c128, %21 : index
      %29 = arith.ceildivui %c2, %24 : index
      %30 = arith.ceildivui %c8, %25 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (4) {
              affine.parallel (%arg12) = (0) to (2) {
                %31 = arith.ceildivui %26, %c2 : index
                scf.for %arg13 = %c0 to %31 step %c1 {
                  %32 = arith.ceildivui %27, %c2 : index
                  scf.for %arg14 = %c0 to %32 step %c1 {
                    %33 = arith.ceildivui %28, %c2 : index
                    scf.for %arg15 = %c0 to %33 step %c1 {
                      %34 = arith.ceildivui %29, %c4 : index
                      scf.for %arg16 = %c0 to %34 step %c1 {
                        %35 = arith.ceildivui %30, %c2 : index
                        scf.for %arg17 = %c0 to %35 step %c1 {
                          %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                          %37 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                          %38 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg10, %arg15)
                          %39 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg11, %arg16)
                          %40 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg12, %arg17)
                          %41 = arith.muli %39, %24 : index
                          %42 = arith.muli %36, %23 : index
                          %43 = arith.muli %40, %25 : index
                          %44 = arith.muli %37, %20 : index
                          %45 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %46 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg12, %c2 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg8, %c4 : index
                          %51 = arith.addi %49, %50 : index
                          %52 = arith.muli %arg11, %c2 : index
                          %53 = arith.addi %52, %c1 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %54 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %55 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %56 = loom.semaphore_take %55 : memref<?x32xf16> -> memref<?x32xf16>
                          %57 = loom.init_tensor %56[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %58 = loom.semaphore_take %55 : memref<?x32xf16> -> memref<?x32xf16>
                          %59 = loom.init_tensor %58[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %60 = loom.broadcast ins(%54 : tensor<?xf16>) outs(%59 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %61 = arith.muli %43, %c512 : index
                          %62 = arith.addi %44, %61 : index
                          %63 = arith.divui %42, %c16 : index
                          %64 = loom.alloc [%20, 128] on @L1 : memref<?x128xf16>
                          %65 = loom.semaphore_take %64 : memref<?x128xf16> -> memref<?x128xf16>
                          %66 = loom.subview %arg4[%41, %62, %63, 0] [1, %20, 1, 128] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x4096x8x128xf16> to memref<?x128xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %66, %65 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?x128xf16, strided<[1024, 1], offset: ?>> to memref<?x128xf16>
                          %67 = loom.bufferize_to_tensor %65[%20, 128] : memref<?x128xf16> -> tensor<?x128xf16>
                          %68 = arith.muli %38, %21 : index
                          %69 = loom.alloc [128, %21] on @L1 : memref<128x?xf16>
                          %70 = loom.semaphore_take %69 : memref<128x?xf16> -> memref<128x?xf16>
                          %71 = loom.subview %arg5[%41, %43, %42, 0, %68] [1, 1, 1, 128, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x128x128x128xf16> to memref<128x?xf16, strided<[128, 1], offset: ?>>
                          %72 = arith.addi %48, %50 : index
                          %73 = arith.addi %48, %c1 : index
                          %74 = arith.addi %73, %50 : index
                          %75 = arith.addi %arg10, %52 : index
                          loom.copy %71, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%72, %75], LR : [%74, %75]) : memref<128x?xf16, strided<[128, 1], offset: ?>> to memref<128x?xf16>
                          %76 = loom.bufferize_to_tensor %70[128, %21] : memref<128x?xf16> -> tensor<128x?xf16>
                          %77 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %79 = loom.init_tensor %78[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %80 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %81 = loom.init_tensor %80[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %82 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %83 = loom.init_tensor %82[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %84 = linalg.fill ins(%cst : f16) outs(%81 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %85 = linalg.matmul ins(%67, %76 : tensor<?x128xf16>, tensor<128x?xf16>) outs(%84 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %70 : memref<128x?xf16>
                          loom.semaphore_give %65 : memref<?x128xf16>
                          %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%85, %60 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%83 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %116 = math.exp %in_0 : f16
                            %117 = arith.mulf %in, %116 : f16
                            linalg.yield %117 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %80 : memref<?x?xf16>
                          loom.semaphore_give %58 : memref<?x32xf16>
                          %87 = arith.addi %37, %c1 : index
                          %88 = arith.muli %87, %20 : index
                          %89 = arith.ceildivui %88, %22 : index
                          %90 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %91 = loom.semaphore_take %90 : memref<32x?xf16> -> memref<32x?xf16>
                          %92 = loom.init_tensor %91[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %93 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %94 = loom.semaphore_take %93 : memref<32x?xf16> -> memref<32x?xf16>
                          %95 = loom.init_tensor %94[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %96 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %97 = loom.semaphore_take %96 : memref<?x?xf16> -> memref<?x?xf16>
                          %98 = loom.init_tensor %97[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %99 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %100 = loom.semaphore_take %99 : memref<?x?xf16> -> memref<?x?xf16>
                          %101 = scf.for %arg18 = %c0 to %89 step %c1 iter_args(%arg19 = %86) -> (tensor<?x?xf16>) {
                            %116 = arith.muli %arg18, %22 : index
                            %117 = arith.addi %116, %22 : index
                            %118 = arith.cmpi ult, %117, %88 : index
                            %119 = arith.select %118, %117, %88 : index
                            %120 = arith.subi %119, %116 : index
                            %121 = loom.alloc [%20, %120] on @L1 : memref<?x?xf16>
                            %122 = loom.semaphore_take %121 : memref<?x?xf16> -> memref<?x?xf16>
                            %123 = loom.subview %arg0[%41, %43, %63, %44, %116] [1, 1, 1, %20, %120] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x8x512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                            loom.copy %123, %122 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                            %124 = loom.bufferize_to_tensor %122[%20, %120] : memref<?x?xf16> -> tensor<?x?xf16>
                            %125 = loom.alloc [%120] on @L1 : memref<?xf16>
                            %126 = loom.semaphore_take %125 : memref<?xf16> -> memref<?xf16>
                            %127 = loom.semaphore_take %125 : memref<?xf16> -> memref<?xf16>
                            %128 = loom.subview %arg1[%41, %42, %43, %116] [1, 1, 1, %120] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %128, %127 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %129 = loom.bufferize_to_tensor %127[%120] : memref<?xf16> -> tensor<?xf16>
                            %130 = loom.broadcast ins(%54 : tensor<?xf16>) outs(%57 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %131 = loom.broadcast ins(%129 : tensor<?xf16>) outs(%92 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %127 : memref<?xf16>
                            %132 = loom.subview %arg2[%41, %42, %43, %116] [1, 1, 1, %120] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %132, %126 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %133 = loom.bufferize_to_tensor %126[%120] : memref<?xf16> -> tensor<?xf16>
                            %134 = loom.broadcast ins(%133 : tensor<?xf16>) outs(%95 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %126 : memref<?xf16>
                            %135 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%124, %130, %131, %134 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%98 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %140 = arith.subf %in_0, %in_1 : f16
                              %141 = math.exp %140 : f16
                              %142 = arith.mulf %in, %141 : f16
                              %143 = arith.mulf %142, %in_2 : f16
                              linalg.yield %143 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %94 : memref<32x?xf16>
                            loom.semaphore_give %91 : memref<32x?xf16>
                            loom.semaphore_give %122 : memref<?x?xf16>
                            loom.semaphore_give %56 : memref<?x32xf16>
                            %136 = arith.addi %116, %61 : index
                            %137 = loom.subview %arg3[%41, %136, %42, %68] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                            loom.copy %137, %100 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?x?xf16, strided<[16384, 1], offset: ?>> to memref<?x?xf16>
                            %138 = loom.bufferize_to_tensor %100[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %139 = linalg.matmul ins(%135, %138 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %100 : memref<?x?xf16>
                            loom.semaphore_give %97 : memref<?x?xf16>
                            scf.yield %139 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %102 = loom.alloc [1] on @L1 : memref<f16>
                          %103 = loom.semaphore_take %102 : memref<f16> -> memref<f16>
                          %104 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<128xf16> to memref<f16, strided<[], offset: ?>>
                          %105 = arith.addi %50, %c3 : index
                          loom.copy %104, %103 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%50, %c0], LR : [%105, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %106 = loom.bufferize_to_tensor %103[] : memref<f16> -> tensor<f16>
                          %107 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %108 = loom.semaphore_take %107 : memref<?x?xf16> -> memref<?x?xf16>
                          %109 = loom.init_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %110 = loom.subview %arg3[%41, %62, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          loom.copy %110, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?x?xf16, strided<[16384, 1], offset: ?>> to memref<?x?xf16>
                          %111 = loom.bufferize_to_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101, %111, %106 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%109 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %116 = arith.mulf %in_0, %in_1 : f16
                            %117 = arith.addf %in, %116 : f16
                            linalg.yield %117 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %103 : memref<f16>
                          loom.semaphore_give %82 : memref<?x?xf16>
                          %113 = loom.sync ins(%112 : tensor<?x?xf16>) outs(%79 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %108 : memref<?x?xf16>
                          %114 = loom.subview %arg7[%41, %62, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          %115 = loom.bufferize_to_memref %113 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %115, %114 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?x?xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          loom.semaphore_give %78 : memref<?x?xf16>
                        } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
                  } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 128 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 128 : index}} {
    func.func @_mamba_chunk_scan__x2x2x2_y4y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc4_dim_y_level0_bc4_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc4_dim_y_level1_bc8_n_n(%arg0: memref<2x8x8x512x512xf16>, %arg1: memref<2x128x8x512xf16>, %arg2: memref<2x128x8x512xf16>, %arg3: memref<2x4096x128x128xf16>, %arg4: memref<2x4096x8x128xf16>, %arg5: memref<2x8x128x128x128xf16>, %arg6: memref<128xf16>, %arg7: memref<2x4096x128x128xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c512 = arith.constant 512 : index
      %c128 = arith.constant 128 : index
      %20 = loom.sym @tile_m {upper_bound = 512 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 128 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 128 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c128, %23 : index
      %27 = arith.ceildivui %c512, %20 : index
      %28 = arith.ceildivui %c128, %21 : index
      %29 = arith.ceildivui %c2, %24 : index
      %30 = arith.ceildivui %c8, %25 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (4) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                %31 = arith.ceildivui %26, %c2 : index
                scf.for %arg13 = %c0 to %31 step %c1 {
                  %32 = arith.ceildivui %27, %c2 : index
                  scf.for %arg14 = %c0 to %32 step %c1 {
                    %33 = arith.ceildivui %28, %c4 : index
                    scf.for %arg15 = %c0 to %33 step %c1 {
                      %34 = arith.ceildivui %29, %c2 : index
                      scf.for %arg16 = %c0 to %34 step %c1 {
                        %35 = arith.ceildivui %30, %c2 : index
                        scf.for %arg17 = %c0 to %35 step %c1 {
                          %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                          %37 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                          %38 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg10, %arg15)
                          %39 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg11, %arg16)
                          %40 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg12, %arg17)
                          %41 = arith.muli %39, %24 : index
                          %42 = arith.muli %36, %23 : index
                          %43 = arith.muli %40, %25 : index
                          %44 = arith.muli %37, %20 : index
                          %45 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %46 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg11, %c2 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg8, %c4 : index
                          %51 = arith.addi %49, %50 : index
                          %52 = arith.muli %arg12, %c4 : index
                          %53 = arith.addi %52, %c3 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %54 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %55 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %56 = loom.semaphore_take %55 : memref<?x32xf16> -> memref<?x32xf16>
                          %57 = loom.init_tensor %56[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %58 = loom.semaphore_take %55 : memref<?x32xf16> -> memref<?x32xf16>
                          %59 = loom.init_tensor %58[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %60 = loom.broadcast ins(%54 : tensor<?xf16>) outs(%59 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %61 = arith.muli %43, %c512 : index
                          %62 = arith.addi %44, %61 : index
                          %63 = arith.divui %42, %c16 : index
                          %64 = loom.alloc [%20, 128] on @L1 : memref<?x128xf16>
                          %65 = loom.semaphore_take %64 : memref<?x128xf16> -> memref<?x128xf16>
                          %66 = loom.subview %arg4[%41, %62, %63, 0] [1, %20, 1, 128] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x4096x8x128xf16> to memref<?x128xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %66, %65 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?x128xf16, strided<[1024, 1], offset: ?>> to memref<?x128xf16>
                          %67 = loom.bufferize_to_tensor %65[%20, 128] : memref<?x128xf16> -> tensor<?x128xf16>
                          %68 = arith.muli %38, %21 : index
                          %69 = loom.alloc [128, %21] on @L1 : memref<128x?xf16>
                          %70 = loom.semaphore_take %69 : memref<128x?xf16> -> memref<128x?xf16>
                          %71 = loom.subview %arg5[%41, %43, %42, 0, %68] [1, 1, 1, 128, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x128x128x128xf16> to memref<128x?xf16, strided<[128, 1], offset: ?>>
                          %72 = arith.addi %48, %50 : index
                          %73 = arith.addi %48, %c1 : index
                          %74 = arith.addi %73, %50 : index
                          %75 = arith.addi %arg10, %52 : index
                          loom.copy %71, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%72, %75], LR : [%74, %75]) : memref<128x?xf16, strided<[128, 1], offset: ?>> to memref<128x?xf16>
                          %76 = loom.bufferize_to_tensor %70[128, %21] : memref<128x?xf16> -> tensor<128x?xf16>
                          %77 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %79 = loom.init_tensor %78[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %80 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %81 = loom.init_tensor %80[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %82 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %83 = loom.init_tensor %82[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %84 = linalg.fill ins(%cst : f16) outs(%81 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %85 = linalg.matmul ins(%67, %76 : tensor<?x128xf16>, tensor<128x?xf16>) outs(%84 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %70 : memref<128x?xf16>
                          loom.semaphore_give %65 : memref<?x128xf16>
                          %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%85, %60 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%83 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %116 = math.exp %in_0 : f16
                            %117 = arith.mulf %in, %116 : f16
                            linalg.yield %117 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %80 : memref<?x?xf16>
                          loom.semaphore_give %58 : memref<?x32xf16>
                          %87 = arith.addi %37, %c1 : index
                          %88 = arith.muli %87, %20 : index
                          %89 = arith.ceildivui %88, %22 : index
                          %90 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %91 = loom.semaphore_take %90 : memref<32x?xf16> -> memref<32x?xf16>
                          %92 = loom.init_tensor %91[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %93 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %94 = loom.semaphore_take %93 : memref<32x?xf16> -> memref<32x?xf16>
                          %95 = loom.init_tensor %94[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %96 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %97 = loom.semaphore_take %96 : memref<?x?xf16> -> memref<?x?xf16>
                          %98 = loom.init_tensor %97[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %99 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %100 = loom.semaphore_take %99 : memref<?x?xf16> -> memref<?x?xf16>
                          %101 = scf.for %arg18 = %c0 to %89 step %c1 iter_args(%arg19 = %86) -> (tensor<?x?xf16>) {
                            %116 = arith.muli %arg18, %22 : index
                            %117 = arith.addi %116, %22 : index
                            %118 = arith.cmpi ult, %117, %88 : index
                            %119 = arith.select %118, %117, %88 : index
                            %120 = arith.subi %119, %116 : index
                            %121 = loom.alloc [%20, %120] on @L1 : memref<?x?xf16>
                            %122 = loom.semaphore_take %121 : memref<?x?xf16> -> memref<?x?xf16>
                            %123 = loom.subview %arg0[%41, %43, %63, %44, %116] [1, 1, 1, %20, %120] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x8x512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                            loom.copy %123, %122 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                            %124 = loom.bufferize_to_tensor %122[%20, %120] : memref<?x?xf16> -> tensor<?x?xf16>
                            %125 = loom.alloc [%120] on @L1 : memref<?xf16>
                            %126 = loom.semaphore_take %125 : memref<?xf16> -> memref<?xf16>
                            %127 = loom.semaphore_take %125 : memref<?xf16> -> memref<?xf16>
                            %128 = loom.subview %arg1[%41, %42, %43, %116] [1, 1, 1, %120] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %128, %127 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %129 = loom.bufferize_to_tensor %127[%120] : memref<?xf16> -> tensor<?xf16>
                            %130 = loom.broadcast ins(%54 : tensor<?xf16>) outs(%57 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %131 = loom.broadcast ins(%129 : tensor<?xf16>) outs(%92 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %127 : memref<?xf16>
                            %132 = loom.subview %arg2[%41, %42, %43, %116] [1, 1, 1, %120] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %132, %126 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %133 = loom.bufferize_to_tensor %126[%120] : memref<?xf16> -> tensor<?xf16>
                            %134 = loom.broadcast ins(%133 : tensor<?xf16>) outs(%95 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %126 : memref<?xf16>
                            %135 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%124, %130, %131, %134 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%98 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %140 = arith.subf %in_0, %in_1 : f16
                              %141 = math.exp %140 : f16
                              %142 = arith.mulf %in, %141 : f16
                              %143 = arith.mulf %142, %in_2 : f16
                              linalg.yield %143 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %94 : memref<32x?xf16>
                            loom.semaphore_give %91 : memref<32x?xf16>
                            loom.semaphore_give %122 : memref<?x?xf16>
                            loom.semaphore_give %56 : memref<?x32xf16>
                            %136 = arith.addi %116, %61 : index
                            %137 = loom.subview %arg3[%41, %136, %42, %68] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                            loom.copy %137, %100 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?x?xf16, strided<[16384, 1], offset: ?>> to memref<?x?xf16>
                            %138 = loom.bufferize_to_tensor %100[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %139 = linalg.matmul ins(%135, %138 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %100 : memref<?x?xf16>
                            loom.semaphore_give %97 : memref<?x?xf16>
                            scf.yield %139 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %102 = loom.alloc [1] on @L1 : memref<f16>
                          %103 = loom.semaphore_take %102 : memref<f16> -> memref<f16>
                          %104 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<128xf16> to memref<f16, strided<[], offset: ?>>
                          %105 = arith.addi %50, %c3 : index
                          loom.copy %104, %103 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%50, %c0], LR : [%105, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %106 = loom.bufferize_to_tensor %103[] : memref<f16> -> tensor<f16>
                          %107 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %108 = loom.semaphore_take %107 : memref<?x?xf16> -> memref<?x?xf16>
                          %109 = loom.init_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %110 = loom.subview %arg3[%41, %62, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          loom.copy %110, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?x?xf16, strided<[16384, 1], offset: ?>> to memref<?x?xf16>
                          %111 = loom.bufferize_to_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101, %111, %106 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%109 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %116 = arith.mulf %in_0, %in_1 : f16
                            %117 = arith.addf %in, %116 : f16
                            linalg.yield %117 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %103 : memref<f16>
                          loom.semaphore_give %82 : memref<?x?xf16>
                          %113 = loom.sync ins(%112 : tensor<?x?xf16>) outs(%79 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %108 : memref<?x?xf16>
                          %114 = loom.subview %arg7[%41, %62, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          %115 = loom.bufferize_to_memref %113 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %115, %114 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?x?xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          loom.semaphore_give %78 : memref<?x?xf16>
                        } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
                  } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 128 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 128 : index}} {
    func.func @_mamba_chunk_scan__x2x2x2_y4y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc4_dim_y_level0_bc4_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc4_dim_y_level1_bc8_n_n(%arg0: memref<2x8x8x512x512xf16>, %arg1: memref<2x128x8x512xf16>, %arg2: memref<2x128x8x512xf16>, %arg3: memref<2x4096x128x128xf16>, %arg4: memref<2x4096x8x128xf16>, %arg5: memref<2x8x128x128x128xf16>, %arg6: memref<128xf16>, %arg7: memref<2x4096x128x128xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c512 = arith.constant 512 : index
      %c128 = arith.constant 128 : index
      %20 = loom.sym @tile_m {upper_bound = 512 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 128 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 128 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c128, %23 : index
      %27 = arith.ceildivui %c512, %20 : index
      %28 = arith.ceildivui %c128, %21 : index
      %29 = arith.ceildivui %c2, %24 : index
      %30 = arith.ceildivui %c8, %25 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (4) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                %31 = arith.ceildivui %26, %c2 : index
                scf.for %arg13 = %c0 to %31 step %c1 {
                  %32 = arith.ceildivui %27, %c2 : index
                  scf.for %arg14 = %c0 to %32 step %c1 {
                    %33 = arith.ceildivui %28, %c4 : index
                    scf.for %arg15 = %c0 to %33 step %c1 {
                      %34 = arith.ceildivui %29, %c2 : index
                      scf.for %arg16 = %c0 to %34 step %c1 {
                        %35 = arith.ceildivui %30, %c2 : index
                        scf.for %arg17 = %c0 to %35 step %c1 {
                          %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                          %37 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                          %38 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg10, %arg15)
                          %39 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg11, %arg16)
                          %40 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg12, %arg17)
                          %41 = arith.muli %39, %24 : index
                          %42 = arith.muli %36, %23 : index
                          %43 = arith.muli %40, %25 : index
                          %44 = arith.muli %37, %20 : index
                          %45 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %46 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg12, %c2 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg8, %c4 : index
                          %51 = arith.addi %49, %50 : index
                          %52 = arith.muli %arg11, %c4 : index
                          %53 = arith.addi %52, %c3 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %54 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %55 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %56 = loom.semaphore_take %55 : memref<?x32xf16> -> memref<?x32xf16>
                          %57 = loom.init_tensor %56[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %58 = loom.semaphore_take %55 : memref<?x32xf16> -> memref<?x32xf16>
                          %59 = loom.init_tensor %58[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %60 = loom.broadcast ins(%54 : tensor<?xf16>) outs(%59 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %61 = arith.muli %43, %c512 : index
                          %62 = arith.addi %44, %61 : index
                          %63 = arith.divui %42, %c16 : index
                          %64 = loom.alloc [%20, 128] on @L1 : memref<?x128xf16>
                          %65 = loom.semaphore_take %64 : memref<?x128xf16> -> memref<?x128xf16>
                          %66 = loom.subview %arg4[%41, %62, %63, 0] [1, %20, 1, 128] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x4096x8x128xf16> to memref<?x128xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %66, %65 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?x128xf16, strided<[1024, 1], offset: ?>> to memref<?x128xf16>
                          %67 = loom.bufferize_to_tensor %65[%20, 128] : memref<?x128xf16> -> tensor<?x128xf16>
                          %68 = arith.muli %38, %21 : index
                          %69 = loom.alloc [128, %21] on @L1 : memref<128x?xf16>
                          %70 = loom.semaphore_take %69 : memref<128x?xf16> -> memref<128x?xf16>
                          %71 = loom.subview %arg5[%41, %43, %42, 0, %68] [1, 1, 1, 128, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x128x128x128xf16> to memref<128x?xf16, strided<[128, 1], offset: ?>>
                          %72 = arith.addi %48, %50 : index
                          %73 = arith.addi %48, %c1 : index
                          %74 = arith.addi %73, %50 : index
                          %75 = arith.addi %arg10, %52 : index
                          loom.copy %71, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%72, %75], LR : [%74, %75]) : memref<128x?xf16, strided<[128, 1], offset: ?>> to memref<128x?xf16>
                          %76 = loom.bufferize_to_tensor %70[128, %21] : memref<128x?xf16> -> tensor<128x?xf16>
                          %77 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %79 = loom.init_tensor %78[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %80 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %81 = loom.init_tensor %80[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %82 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %83 = loom.init_tensor %82[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %84 = linalg.fill ins(%cst : f16) outs(%81 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %85 = linalg.matmul ins(%67, %76 : tensor<?x128xf16>, tensor<128x?xf16>) outs(%84 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %70 : memref<128x?xf16>
                          loom.semaphore_give %65 : memref<?x128xf16>
                          %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%85, %60 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%83 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %116 = math.exp %in_0 : f16
                            %117 = arith.mulf %in, %116 : f16
                            linalg.yield %117 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %80 : memref<?x?xf16>
                          loom.semaphore_give %58 : memref<?x32xf16>
                          %87 = arith.addi %37, %c1 : index
                          %88 = arith.muli %87, %20 : index
                          %89 = arith.ceildivui %88, %22 : index
                          %90 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %91 = loom.semaphore_take %90 : memref<32x?xf16> -> memref<32x?xf16>
                          %92 = loom.init_tensor %91[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %93 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %94 = loom.semaphore_take %93 : memref<32x?xf16> -> memref<32x?xf16>
                          %95 = loom.init_tensor %94[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %96 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %97 = loom.semaphore_take %96 : memref<?x?xf16> -> memref<?x?xf16>
                          %98 = loom.init_tensor %97[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %99 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %100 = loom.semaphore_take %99 : memref<?x?xf16> -> memref<?x?xf16>
                          %101 = scf.for %arg18 = %c0 to %89 step %c1 iter_args(%arg19 = %86) -> (tensor<?x?xf16>) {
                            %116 = arith.muli %arg18, %22 : index
                            %117 = arith.addi %116, %22 : index
                            %118 = arith.cmpi ult, %117, %88 : index
                            %119 = arith.select %118, %117, %88 : index
                            %120 = arith.subi %119, %116 : index
                            %121 = loom.alloc [%20, %120] on @L1 : memref<?x?xf16>
                            %122 = loom.semaphore_take %121 : memref<?x?xf16> -> memref<?x?xf16>
                            %123 = loom.subview %arg0[%41, %43, %63, %44, %116] [1, 1, 1, %20, %120] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x8x512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                            loom.copy %123, %122 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                            %124 = loom.bufferize_to_tensor %122[%20, %120] : memref<?x?xf16> -> tensor<?x?xf16>
                            %125 = loom.alloc [%120] on @L1 : memref<?xf16>
                            %126 = loom.semaphore_take %125 : memref<?xf16> -> memref<?xf16>
                            %127 = loom.semaphore_take %125 : memref<?xf16> -> memref<?xf16>
                            %128 = loom.subview %arg1[%41, %42, %43, %116] [1, 1, 1, %120] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %128, %127 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %129 = loom.bufferize_to_tensor %127[%120] : memref<?xf16> -> tensor<?xf16>
                            %130 = loom.broadcast ins(%54 : tensor<?xf16>) outs(%57 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %131 = loom.broadcast ins(%129 : tensor<?xf16>) outs(%92 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %127 : memref<?xf16>
                            %132 = loom.subview %arg2[%41, %42, %43, %116] [1, 1, 1, %120] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %132, %126 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %133 = loom.bufferize_to_tensor %126[%120] : memref<?xf16> -> tensor<?xf16>
                            %134 = loom.broadcast ins(%133 : tensor<?xf16>) outs(%95 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %126 : memref<?xf16>
                            %135 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%124, %130, %131, %134 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%98 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %140 = arith.subf %in_0, %in_1 : f16
                              %141 = math.exp %140 : f16
                              %142 = arith.mulf %in, %141 : f16
                              %143 = arith.mulf %142, %in_2 : f16
                              linalg.yield %143 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %94 : memref<32x?xf16>
                            loom.semaphore_give %91 : memref<32x?xf16>
                            loom.semaphore_give %122 : memref<?x?xf16>
                            loom.semaphore_give %56 : memref<?x32xf16>
                            %136 = arith.addi %116, %61 : index
                            %137 = loom.subview %arg3[%41, %136, %42, %68] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                            loom.copy %137, %100 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?x?xf16, strided<[16384, 1], offset: ?>> to memref<?x?xf16>
                            %138 = loom.bufferize_to_tensor %100[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %139 = linalg.matmul ins(%135, %138 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %100 : memref<?x?xf16>
                            loom.semaphore_give %97 : memref<?x?xf16>
                            scf.yield %139 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %102 = loom.alloc [1] on @L1 : memref<f16>
                          %103 = loom.semaphore_take %102 : memref<f16> -> memref<f16>
                          %104 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<128xf16> to memref<f16, strided<[], offset: ?>>
                          %105 = arith.addi %50, %c3 : index
                          loom.copy %104, %103 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%50, %c0], LR : [%105, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %106 = loom.bufferize_to_tensor %103[] : memref<f16> -> tensor<f16>
                          %107 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %108 = loom.semaphore_take %107 : memref<?x?xf16> -> memref<?x?xf16>
                          %109 = loom.init_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %110 = loom.subview %arg3[%41, %62, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          loom.copy %110, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?x?xf16, strided<[16384, 1], offset: ?>> to memref<?x?xf16>
                          %111 = loom.bufferize_to_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101, %111, %106 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%109 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %116 = arith.mulf %in_0, %in_1 : f16
                            %117 = arith.addf %in, %116 : f16
                            linalg.yield %117 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %103 : memref<f16>
                          loom.semaphore_give %82 : memref<?x?xf16>
                          %113 = loom.sync ins(%112 : tensor<?x?xf16>) outs(%79 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %108 : memref<?x?xf16>
                          %114 = loom.subview %arg7[%41, %62, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          %115 = loom.bufferize_to_memref %113 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %115, %114 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%51, %75], LR : [%51, %75]) : memref<?x?xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
                          loom.semaphore_give %78 : memref<?x?xf16>
                        } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
                  } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
