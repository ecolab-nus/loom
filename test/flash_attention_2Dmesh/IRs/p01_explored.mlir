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
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_flash__attention__x8_y8__d1i0_d0i1__f01__n_n_n_n(%arg0: memref<256x128x4096xf16>, %arg1: memref<256x4096x128xf16>, %arg2: memref<256x4096x128xf16>, %arg3: memref<256x4096x128xf16>) {
      %c8 = arith.constant 8 : index
      %cst = arith.constant 2.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.000000e+00 : f16
      %cst_2 = arith.constant 0xFC00 : f16
      %cst_3 = arith.constant 1.275630e-01 : f16
      %c256 = arith.constant 256 : index
      %c4096 = arith.constant 4096 : index
      %20 = loom.sym @tile_b {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c4096, %21 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          %25 = arith.ceildivui %23, %c8 : index
          scf.for %arg6 = %c0 to %25 step %c1 {
            %26 = arith.ceildivui %24, %c8 : index
            scf.for %arg7 = %c0 to %26 step %c1 {
              %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %28 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
              %29 = arith.muli %27, %20 : index
              %30 = arith.muli %28, %21 : index
              %31 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %32 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %33 = loom.init_tensor %32[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %34 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %35 = loom.subview %arg2[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
              %36 = loom.bufferize_to_tensor %34[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %37 = arith.ceildivui %c4096, %22 : index
              %38 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %39 = loom.semaphore_take %38 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %40 = loom.init_tensor %39[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
              %42 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
              %44 = loom.init_tensor %43[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %45 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %46 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
              %48 = loom.init_tensor %47[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %49 = linalg.fill ins(%cst_2 : f16) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %50 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %51 = loom.semaphore_take %50 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %52 = loom.init_tensor %51[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %53 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %54 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
              %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %56 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %57 = loom.semaphore_take %56 : memref<?x?xf16> -> memref<?x?xf16>
              %58 = loom.init_tensor %57[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %59 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %60 = loom.semaphore_take %59 : memref<?x?xf16> -> memref<?x?xf16>
              %61 = loom.init_tensor %60[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %62 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
              %63 = loom.semaphore_take %62 : memref<?x128x?xf16> -> memref<?x128x?xf16>
              %64 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
              %65 = loom.semaphore_take %64 : memref<?x?x?xf16> -> memref<?x?x?xf16>
              %66 = loom.init_tensor %65[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
              %67 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
              %68 = loom.semaphore_take %67 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %69:3 = scf.for %arg8 = %c0 to %37 step %c1 iter_args(%arg9 = %49, %arg10 = %45, %arg11 = %41) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>) {
                %73 = arith.muli %arg8, %22 : index
                %74 = loom.subview %arg0[%29, 0, %73] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                loom.copy %74, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                %75 = loom.bufferize_to_tensor %63[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                %76 = linalg.fill ins(%cst_0 : f16) outs(%66 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                %77 = linalg.batch_matmul ins(%36, %75 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%76 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                loom.semaphore_give %63 : memref<?x128x?xf16>
                %78 = linalg.fill ins(%cst_2 : f16) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%77 : tensor<?x?x?xf16>) outs(%78 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.maximumf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<?x?xf16>
                %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %79 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.mulf %in_4, %cst_3 : f16
                  %93 = arith.cmpf ogt, %in, %92 : f16
                  %94 = arith.select %93, %in, %92 : f16
                  linalg.yield %94 : f16
                } -> tensor<?x?xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%77, %80 : tensor<?x?x?xf16>, tensor<?x?xf16>) outs(%66 : tensor<?x?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.mulf %in, %cst_3 : f16
                  %93 = arith.subf %92, %in_4 : f16
                  %94 = math.powf %cst, %93 : f16
                  linalg.yield %94 : f16
                } -> tensor<?x?x?xf16>
                %82 = linalg.fill ins(%cst_0 : f16) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%81 : tensor<?x?x?xf16>) outs(%82 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.addf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<?x?xf16>
                %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %80 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%61 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.subf %in, %in_4 : f16
                  %93 = math.powf %cst, %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %84, %83 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg10 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
                  %92 = arith.mulf %in, %in_4 : f16
                  %93 = arith.addf %92, %in_5 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?xf16>
                loom.semaphore_give %57 : memref<?x?xf16>
                %86 = loom.subview %arg1[%29, %73, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %86, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %87 = loom.bufferize_to_tensor %68[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %88 = linalg.fill ins(%cst_0 : f16) outs(%52 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %89 = linalg.batch_matmul ins(%81, %87 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%88 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %68 : memref<?x?x128xf16>
                loom.semaphore_give %65 : memref<?x?x?xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %arg11, %84 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?xf16>) outs(%arg11 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
                  %92 = arith.mulf %in_4, %in_5 : f16
                  %93 = arith.addf %in, %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %60 : memref<?x?xf16>
                loom.semaphore_give %51 : memref<?x?x128xf16>
                %91 = linalg.copy ins(%80 : tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %54 : memref<?x?xf16>
                scf.yield %91, %85, %90 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              loom.semaphore_give %47 : memref<?x?xf16>
              loom.semaphore_give %34 : memref<?x?x128xf16>
              %70 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%69#2, %69#1 : tensor<?x?x128xf16>, tensor<?x?xf16>) outs(%33 : tensor<?x?x128xf16>) {
              ^bb0(%in: f16, %in_4: f16, %out: f16):
                %73 = arith.divf %in, %in_4 : f16
                linalg.yield %73 : f16
              } -> tensor<?x?x128xf16>
              loom.semaphore_give %43 : memref<?x?xf16>
              loom.semaphore_give %39 : memref<?x?x128xf16>
              %71 = loom.subview %arg3[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              %72 = loom.bufferize_to_memref %70 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
              loom.copy %72, %71 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %32 : memref<?x?x128xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_flash__attention__x8_y8__d1i0_d0i1__f01__n_n_dim_x_level0_bc8_n(%arg0: memref<256x128x4096xf16>, %arg1: memref<256x4096x128xf16>, %arg2: memref<256x4096x128xf16>, %arg3: memref<256x4096x128xf16>) {
      %c8 = arith.constant 8 : index
      %cst = arith.constant 2.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.000000e+00 : f16
      %cst_2 = arith.constant 0xFC00 : f16
      %cst_3 = arith.constant 1.275630e-01 : f16
      %c256 = arith.constant 256 : index
      %c4096 = arith.constant 4096 : index
      %20 = loom.sym @tile_b {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c4096, %21 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          %25 = arith.ceildivui %23, %c8 : index
          scf.for %arg6 = %c0 to %25 step %c1 {
            %26 = arith.ceildivui %24, %c8 : index
            scf.for %arg7 = %c0 to %26 step %c1 {
              %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %28 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
              %29 = arith.muli %27, %20 : index
              %30 = arith.muli %28, %21 : index
              %31 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %32 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %33 = loom.init_tensor %32[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %34 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %35 = loom.subview %arg2[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
              %36 = loom.bufferize_to_tensor %34[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %37 = arith.ceildivui %c4096, %22 : index
              %38 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %39 = loom.semaphore_take %38 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %40 = loom.init_tensor %39[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
              %42 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
              %44 = loom.init_tensor %43[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %45 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %46 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
              %48 = loom.init_tensor %47[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %49 = linalg.fill ins(%cst_2 : f16) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %50 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %51 = loom.semaphore_take %50 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %52 = loom.init_tensor %51[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %53 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %54 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
              %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %56 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %57 = loom.semaphore_take %56 : memref<?x?xf16> -> memref<?x?xf16>
              %58 = loom.init_tensor %57[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %59 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %60 = loom.semaphore_take %59 : memref<?x?xf16> -> memref<?x?xf16>
              %61 = loom.init_tensor %60[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %62 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
              %63 = loom.semaphore_take %62 : memref<?x128x?xf16> -> memref<?x128x?xf16>
              %64 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
              %65 = loom.semaphore_take %64 : memref<?x?x?xf16> -> memref<?x?x?xf16>
              %66 = loom.init_tensor %65[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
              %67 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
              %68 = loom.semaphore_take %67 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %69:3 = scf.for %arg8 = %c0 to %37 step %c1 iter_args(%arg9 = %49, %arg10 = %45, %arg11 = %41) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>) {
                %73 = arith.muli %arg8, %22 : index
                %74 = loom.subview %arg0[%29, 0, %73] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                loom.copy %74, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                %75 = loom.bufferize_to_tensor %63[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                %76 = linalg.fill ins(%cst_0 : f16) outs(%66 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                %77 = linalg.batch_matmul ins(%36, %75 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%76 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                loom.semaphore_give %63 : memref<?x128x?xf16>
                %78 = linalg.fill ins(%cst_2 : f16) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%77 : tensor<?x?x?xf16>) outs(%78 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.maximumf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<?x?xf16>
                %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %79 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.mulf %in_4, %cst_3 : f16
                  %93 = arith.cmpf ogt, %in, %92 : f16
                  %94 = arith.select %93, %in, %92 : f16
                  linalg.yield %94 : f16
                } -> tensor<?x?xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%77, %80 : tensor<?x?x?xf16>, tensor<?x?xf16>) outs(%66 : tensor<?x?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.mulf %in, %cst_3 : f16
                  %93 = arith.subf %92, %in_4 : f16
                  %94 = math.powf %cst, %93 : f16
                  linalg.yield %94 : f16
                } -> tensor<?x?x?xf16>
                %82 = linalg.fill ins(%cst_0 : f16) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%81 : tensor<?x?x?xf16>) outs(%82 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.addf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<?x?xf16>
                %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %80 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%61 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.subf %in, %in_4 : f16
                  %93 = math.powf %cst, %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %84, %83 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg10 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
                  %92 = arith.mulf %in, %in_4 : f16
                  %93 = arith.addf %92, %in_5 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?xf16>
                loom.semaphore_give %57 : memref<?x?xf16>
                %86 = loom.subview %arg1[%29, %73, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %86, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %87 = loom.bufferize_to_tensor %68[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %88 = linalg.fill ins(%cst_0 : f16) outs(%52 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %89 = linalg.batch_matmul ins(%81, %87 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%88 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %68 : memref<?x?x128xf16>
                loom.semaphore_give %65 : memref<?x?x?xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %arg11, %84 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?xf16>) outs(%arg11 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
                  %92 = arith.mulf %in_4, %in_5 : f16
                  %93 = arith.addf %in, %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %60 : memref<?x?xf16>
                loom.semaphore_give %51 : memref<?x?x128xf16>
                %91 = linalg.copy ins(%80 : tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %54 : memref<?x?xf16>
                scf.yield %91, %85, %90 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              loom.semaphore_give %47 : memref<?x?xf16>
              loom.semaphore_give %34 : memref<?x?x128xf16>
              %70 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%69#2, %69#1 : tensor<?x?x128xf16>, tensor<?x?xf16>) outs(%33 : tensor<?x?x128xf16>) {
              ^bb0(%in: f16, %in_4: f16, %out: f16):
                %73 = arith.divf %in, %in_4 : f16
                linalg.yield %73 : f16
              } -> tensor<?x?x128xf16>
              loom.semaphore_give %43 : memref<?x?xf16>
              loom.semaphore_give %39 : memref<?x?x128xf16>
              %71 = loom.subview %arg3[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              %72 = loom.bufferize_to_memref %70 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
              loom.copy %72, %71 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %32 : memref<?x?x128xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_flash__attention__x8_y8__d1i0_d0i1__f01__n_dim_x_level0_bc8_n_n(%arg0: memref<256x128x4096xf16>, %arg1: memref<256x4096x128xf16>, %arg2: memref<256x4096x128xf16>, %arg3: memref<256x4096x128xf16>) {
      %c8 = arith.constant 8 : index
      %cst = arith.constant 2.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.000000e+00 : f16
      %cst_2 = arith.constant 0xFC00 : f16
      %cst_3 = arith.constant 1.275630e-01 : f16
      %c256 = arith.constant 256 : index
      %c4096 = arith.constant 4096 : index
      %20 = loom.sym @tile_b {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c4096, %21 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          %25 = arith.ceildivui %23, %c8 : index
          scf.for %arg6 = %c0 to %25 step %c1 {
            %26 = arith.ceildivui %24, %c8 : index
            scf.for %arg7 = %c0 to %26 step %c1 {
              %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %28 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
              %29 = arith.muli %27, %20 : index
              %30 = arith.muli %28, %21 : index
              %31 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %32 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %33 = loom.init_tensor %32[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %34 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %35 = loom.subview %arg2[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
              %36 = loom.bufferize_to_tensor %34[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %37 = arith.ceildivui %c4096, %22 : index
              %38 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %39 = loom.semaphore_take %38 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %40 = loom.init_tensor %39[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
              %42 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
              %44 = loom.init_tensor %43[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %45 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %46 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
              %48 = loom.init_tensor %47[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %49 = linalg.fill ins(%cst_2 : f16) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %50 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %51 = loom.semaphore_take %50 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %52 = loom.init_tensor %51[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %53 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %54 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
              %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %56 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %57 = loom.semaphore_take %56 : memref<?x?xf16> -> memref<?x?xf16>
              %58 = loom.init_tensor %57[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %59 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %60 = loom.semaphore_take %59 : memref<?x?xf16> -> memref<?x?xf16>
              %61 = loom.init_tensor %60[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %62 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
              %63 = loom.semaphore_take %62 : memref<?x128x?xf16> -> memref<?x128x?xf16>
              %64 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
              %65 = loom.semaphore_take %64 : memref<?x?x?xf16> -> memref<?x?x?xf16>
              %66 = loom.init_tensor %65[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
              %67 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
              %68 = loom.semaphore_take %67 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %69:3 = scf.for %arg8 = %c0 to %37 step %c1 iter_args(%arg9 = %49, %arg10 = %45, %arg11 = %41) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>) {
                %73 = arith.muli %arg8, %22 : index
                %74 = loom.subview %arg0[%29, 0, %73] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                loom.copy %74, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                %75 = loom.bufferize_to_tensor %63[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                %76 = linalg.fill ins(%cst_0 : f16) outs(%66 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                %77 = linalg.batch_matmul ins(%36, %75 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%76 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                loom.semaphore_give %63 : memref<?x128x?xf16>
                %78 = linalg.fill ins(%cst_2 : f16) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%77 : tensor<?x?x?xf16>) outs(%78 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.maximumf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<?x?xf16>
                %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %79 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.mulf %in_4, %cst_3 : f16
                  %93 = arith.cmpf ogt, %in, %92 : f16
                  %94 = arith.select %93, %in, %92 : f16
                  linalg.yield %94 : f16
                } -> tensor<?x?xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%77, %80 : tensor<?x?x?xf16>, tensor<?x?xf16>) outs(%66 : tensor<?x?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.mulf %in, %cst_3 : f16
                  %93 = arith.subf %92, %in_4 : f16
                  %94 = math.powf %cst, %93 : f16
                  linalg.yield %94 : f16
                } -> tensor<?x?x?xf16>
                %82 = linalg.fill ins(%cst_0 : f16) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%81 : tensor<?x?x?xf16>) outs(%82 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.addf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<?x?xf16>
                %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %80 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%61 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.subf %in, %in_4 : f16
                  %93 = math.powf %cst, %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %84, %83 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg10 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
                  %92 = arith.mulf %in, %in_4 : f16
                  %93 = arith.addf %92, %in_5 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?xf16>
                loom.semaphore_give %57 : memref<?x?xf16>
                %86 = loom.subview %arg1[%29, %73, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %86, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %87 = loom.bufferize_to_tensor %68[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %88 = linalg.fill ins(%cst_0 : f16) outs(%52 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %89 = linalg.batch_matmul ins(%81, %87 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%88 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %68 : memref<?x?x128xf16>
                loom.semaphore_give %65 : memref<?x?x?xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %arg11, %84 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?xf16>) outs(%arg11 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
                  %92 = arith.mulf %in_4, %in_5 : f16
                  %93 = arith.addf %in, %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %60 : memref<?x?xf16>
                loom.semaphore_give %51 : memref<?x?x128xf16>
                %91 = linalg.copy ins(%80 : tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %54 : memref<?x?xf16>
                scf.yield %91, %85, %90 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              loom.semaphore_give %47 : memref<?x?xf16>
              loom.semaphore_give %34 : memref<?x?x128xf16>
              %70 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%69#2, %69#1 : tensor<?x?x128xf16>, tensor<?x?xf16>) outs(%33 : tensor<?x?x128xf16>) {
              ^bb0(%in: f16, %in_4: f16, %out: f16):
                %73 = arith.divf %in, %in_4 : f16
                linalg.yield %73 : f16
              } -> tensor<?x?x128xf16>
              loom.semaphore_give %43 : memref<?x?xf16>
              loom.semaphore_give %39 : memref<?x?x128xf16>
              %71 = loom.subview %arg3[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              %72 = loom.bufferize_to_memref %70 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
              loom.copy %72, %71 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %32 : memref<?x?x128xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_flash__attention__x8_y8__d1i0_d0i1__f01__n_dim_x_level0_bc8_dim_x_level0_bc8_n(%arg0: memref<256x128x4096xf16>, %arg1: memref<256x4096x128xf16>, %arg2: memref<256x4096x128xf16>, %arg3: memref<256x4096x128xf16>) {
      %c8 = arith.constant 8 : index
      %cst = arith.constant 2.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.000000e+00 : f16
      %cst_2 = arith.constant 0xFC00 : f16
      %cst_3 = arith.constant 1.275630e-01 : f16
      %c256 = arith.constant 256 : index
      %c4096 = arith.constant 4096 : index
      %20 = loom.sym @tile_b {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c4096, %21 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          %25 = arith.ceildivui %23, %c8 : index
          scf.for %arg6 = %c0 to %25 step %c1 {
            %26 = arith.ceildivui %24, %c8 : index
            scf.for %arg7 = %c0 to %26 step %c1 {
              %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %28 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
              %29 = arith.muli %27, %20 : index
              %30 = arith.muli %28, %21 : index
              %31 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %32 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %33 = loom.init_tensor %32[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %34 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %35 = loom.subview %arg2[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
              %36 = loom.bufferize_to_tensor %34[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %37 = arith.ceildivui %c4096, %22 : index
              %38 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %39 = loom.semaphore_take %38 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %40 = loom.init_tensor %39[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
              %42 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
              %44 = loom.init_tensor %43[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %45 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %46 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
              %48 = loom.init_tensor %47[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %49 = linalg.fill ins(%cst_2 : f16) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %50 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %51 = loom.semaphore_take %50 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %52 = loom.init_tensor %51[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %53 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %54 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
              %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %56 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %57 = loom.semaphore_take %56 : memref<?x?xf16> -> memref<?x?xf16>
              %58 = loom.init_tensor %57[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %59 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %60 = loom.semaphore_take %59 : memref<?x?xf16> -> memref<?x?xf16>
              %61 = loom.init_tensor %60[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %62 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
              %63 = loom.semaphore_take %62 : memref<?x128x?xf16> -> memref<?x128x?xf16>
              %64 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
              %65 = loom.semaphore_take %64 : memref<?x?x?xf16> -> memref<?x?x?xf16>
              %66 = loom.init_tensor %65[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
              %67 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
              %68 = loom.semaphore_take %67 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %69:3 = scf.for %arg8 = %c0 to %37 step %c1 iter_args(%arg9 = %49, %arg10 = %45, %arg11 = %41) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>) {
                %73 = arith.muli %arg8, %22 : index
                %74 = loom.subview %arg0[%29, 0, %73] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                loom.copy %74, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                %75 = loom.bufferize_to_tensor %63[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                %76 = linalg.fill ins(%cst_0 : f16) outs(%66 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                %77 = linalg.batch_matmul ins(%36, %75 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%76 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                loom.semaphore_give %63 : memref<?x128x?xf16>
                %78 = linalg.fill ins(%cst_2 : f16) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%77 : tensor<?x?x?xf16>) outs(%78 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.maximumf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<?x?xf16>
                %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %79 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.mulf %in_4, %cst_3 : f16
                  %93 = arith.cmpf ogt, %in, %92 : f16
                  %94 = arith.select %93, %in, %92 : f16
                  linalg.yield %94 : f16
                } -> tensor<?x?xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%77, %80 : tensor<?x?x?xf16>, tensor<?x?xf16>) outs(%66 : tensor<?x?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.mulf %in, %cst_3 : f16
                  %93 = arith.subf %92, %in_4 : f16
                  %94 = math.powf %cst, %93 : f16
                  linalg.yield %94 : f16
                } -> tensor<?x?x?xf16>
                %82 = linalg.fill ins(%cst_0 : f16) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%81 : tensor<?x?x?xf16>) outs(%82 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.addf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<?x?xf16>
                %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %80 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%61 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.subf %in, %in_4 : f16
                  %93 = math.powf %cst, %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %84, %83 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg10 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
                  %92 = arith.mulf %in, %in_4 : f16
                  %93 = arith.addf %92, %in_5 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?xf16>
                loom.semaphore_give %57 : memref<?x?xf16>
                %86 = loom.subview %arg1[%29, %73, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %86, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %87 = loom.bufferize_to_tensor %68[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %88 = linalg.fill ins(%cst_0 : f16) outs(%52 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %89 = linalg.batch_matmul ins(%81, %87 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%88 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %68 : memref<?x?x128xf16>
                loom.semaphore_give %65 : memref<?x?x?xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %arg11, %84 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?xf16>) outs(%arg11 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
                  %92 = arith.mulf %in_4, %in_5 : f16
                  %93 = arith.addf %in, %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %60 : memref<?x?xf16>
                loom.semaphore_give %51 : memref<?x?x128xf16>
                %91 = linalg.copy ins(%80 : tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %54 : memref<?x?xf16>
                scf.yield %91, %85, %90 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              loom.semaphore_give %47 : memref<?x?xf16>
              loom.semaphore_give %34 : memref<?x?x128xf16>
              %70 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%69#2, %69#1 : tensor<?x?x128xf16>, tensor<?x?xf16>) outs(%33 : tensor<?x?x128xf16>) {
              ^bb0(%in: f16, %in_4: f16, %out: f16):
                %73 = arith.divf %in, %in_4 : f16
                linalg.yield %73 : f16
              } -> tensor<?x?x128xf16>
              loom.semaphore_give %43 : memref<?x?xf16>
              loom.semaphore_give %39 : memref<?x?x128xf16>
              %71 = loom.subview %arg3[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              %72 = loom.bufferize_to_memref %70 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
              loom.copy %72, %71 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %32 : memref<?x?x128xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_flash__attention__x8_y8__d1i0_d0i1__f10__n_n_n_n(%arg0: memref<256x128x4096xf16>, %arg1: memref<256x4096x128xf16>, %arg2: memref<256x4096x128xf16>, %arg3: memref<256x4096x128xf16>) {
      %c8 = arith.constant 8 : index
      %cst = arith.constant 2.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.000000e+00 : f16
      %cst_2 = arith.constant 0xFC00 : f16
      %cst_3 = arith.constant 1.275630e-01 : f16
      %c256 = arith.constant 256 : index
      %c4096 = arith.constant 4096 : index
      %20 = loom.sym @tile_b {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c4096, %21 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          %25 = arith.ceildivui %24, %c8 : index
          scf.for %arg6 = %c0 to %25 step %c1 {
            %26 = arith.ceildivui %23, %c8 : index
            scf.for %arg7 = %c0 to %26 step %c1 {
              %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
              %28 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
              %29 = arith.muli %27, %20 : index
              %30 = arith.muli %28, %21 : index
              %31 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %32 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %33 = loom.init_tensor %32[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %34 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %35 = loom.subview %arg2[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
              %36 = loom.bufferize_to_tensor %34[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %37 = arith.ceildivui %c4096, %22 : index
              %38 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %39 = loom.semaphore_take %38 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %40 = loom.init_tensor %39[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
              %42 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
              %44 = loom.init_tensor %43[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %45 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %46 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
              %48 = loom.init_tensor %47[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %49 = linalg.fill ins(%cst_2 : f16) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %50 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %51 = loom.semaphore_take %50 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %52 = loom.init_tensor %51[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %53 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %54 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
              %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %56 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %57 = loom.semaphore_take %56 : memref<?x?xf16> -> memref<?x?xf16>
              %58 = loom.init_tensor %57[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %59 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %60 = loom.semaphore_take %59 : memref<?x?xf16> -> memref<?x?xf16>
              %61 = loom.init_tensor %60[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %62 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
              %63 = loom.semaphore_take %62 : memref<?x128x?xf16> -> memref<?x128x?xf16>
              %64 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
              %65 = loom.semaphore_take %64 : memref<?x?x?xf16> -> memref<?x?x?xf16>
              %66 = loom.init_tensor %65[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
              %67 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
              %68 = loom.semaphore_take %67 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %69:3 = scf.for %arg8 = %c0 to %37 step %c1 iter_args(%arg9 = %49, %arg10 = %45, %arg11 = %41) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>) {
                %73 = arith.muli %arg8, %22 : index
                %74 = loom.subview %arg0[%29, 0, %73] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                loom.copy %74, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                %75 = loom.bufferize_to_tensor %63[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                %76 = linalg.fill ins(%cst_0 : f16) outs(%66 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                %77 = linalg.batch_matmul ins(%36, %75 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%76 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                loom.semaphore_give %63 : memref<?x128x?xf16>
                %78 = linalg.fill ins(%cst_2 : f16) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%77 : tensor<?x?x?xf16>) outs(%78 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.maximumf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<?x?xf16>
                %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %79 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.mulf %in_4, %cst_3 : f16
                  %93 = arith.cmpf ogt, %in, %92 : f16
                  %94 = arith.select %93, %in, %92 : f16
                  linalg.yield %94 : f16
                } -> tensor<?x?xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%77, %80 : tensor<?x?x?xf16>, tensor<?x?xf16>) outs(%66 : tensor<?x?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.mulf %in, %cst_3 : f16
                  %93 = arith.subf %92, %in_4 : f16
                  %94 = math.powf %cst, %93 : f16
                  linalg.yield %94 : f16
                } -> tensor<?x?x?xf16>
                %82 = linalg.fill ins(%cst_0 : f16) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%81 : tensor<?x?x?xf16>) outs(%82 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.addf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<?x?xf16>
                %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %80 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%61 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.subf %in, %in_4 : f16
                  %93 = math.powf %cst, %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %84, %83 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg10 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
                  %92 = arith.mulf %in, %in_4 : f16
                  %93 = arith.addf %92, %in_5 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?xf16>
                loom.semaphore_give %57 : memref<?x?xf16>
                %86 = loom.subview %arg1[%29, %73, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %86, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %87 = loom.bufferize_to_tensor %68[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %88 = linalg.fill ins(%cst_0 : f16) outs(%52 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %89 = linalg.batch_matmul ins(%81, %87 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%88 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %68 : memref<?x?x128xf16>
                loom.semaphore_give %65 : memref<?x?x?xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %arg11, %84 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?xf16>) outs(%arg11 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
                  %92 = arith.mulf %in_4, %in_5 : f16
                  %93 = arith.addf %in, %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %60 : memref<?x?xf16>
                loom.semaphore_give %51 : memref<?x?x128xf16>
                %91 = linalg.copy ins(%80 : tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %54 : memref<?x?xf16>
                scf.yield %91, %85, %90 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              loom.semaphore_give %47 : memref<?x?xf16>
              loom.semaphore_give %34 : memref<?x?x128xf16>
              %70 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%69#2, %69#1 : tensor<?x?x128xf16>, tensor<?x?xf16>) outs(%33 : tensor<?x?x128xf16>) {
              ^bb0(%in: f16, %in_4: f16, %out: f16):
                %73 = arith.divf %in, %in_4 : f16
                linalg.yield %73 : f16
              } -> tensor<?x?x128xf16>
              loom.semaphore_give %43 : memref<?x?xf16>
              loom.semaphore_give %39 : memref<?x?x128xf16>
              %71 = loom.subview %arg3[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              %72 = loom.bufferize_to_memref %70 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
              loom.copy %72, %71 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %32 : memref<?x?x128xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_flash__attention__x8_y8__d1i0_d0i1__f10__n_n_dim_x_level0_bc8_n(%arg0: memref<256x128x4096xf16>, %arg1: memref<256x4096x128xf16>, %arg2: memref<256x4096x128xf16>, %arg3: memref<256x4096x128xf16>) {
      %c8 = arith.constant 8 : index
      %cst = arith.constant 2.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.000000e+00 : f16
      %cst_2 = arith.constant 0xFC00 : f16
      %cst_3 = arith.constant 1.275630e-01 : f16
      %c256 = arith.constant 256 : index
      %c4096 = arith.constant 4096 : index
      %20 = loom.sym @tile_b {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c4096, %21 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          %25 = arith.ceildivui %24, %c8 : index
          scf.for %arg6 = %c0 to %25 step %c1 {
            %26 = arith.ceildivui %23, %c8 : index
            scf.for %arg7 = %c0 to %26 step %c1 {
              %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
              %28 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
              %29 = arith.muli %27, %20 : index
              %30 = arith.muli %28, %21 : index
              %31 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %32 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %33 = loom.init_tensor %32[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %34 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %35 = loom.subview %arg2[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
              %36 = loom.bufferize_to_tensor %34[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %37 = arith.ceildivui %c4096, %22 : index
              %38 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %39 = loom.semaphore_take %38 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %40 = loom.init_tensor %39[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
              %42 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
              %44 = loom.init_tensor %43[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %45 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %46 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
              %48 = loom.init_tensor %47[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %49 = linalg.fill ins(%cst_2 : f16) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %50 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %51 = loom.semaphore_take %50 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %52 = loom.init_tensor %51[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %53 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %54 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
              %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %56 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %57 = loom.semaphore_take %56 : memref<?x?xf16> -> memref<?x?xf16>
              %58 = loom.init_tensor %57[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %59 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %60 = loom.semaphore_take %59 : memref<?x?xf16> -> memref<?x?xf16>
              %61 = loom.init_tensor %60[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %62 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
              %63 = loom.semaphore_take %62 : memref<?x128x?xf16> -> memref<?x128x?xf16>
              %64 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
              %65 = loom.semaphore_take %64 : memref<?x?x?xf16> -> memref<?x?x?xf16>
              %66 = loom.init_tensor %65[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
              %67 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
              %68 = loom.semaphore_take %67 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %69:3 = scf.for %arg8 = %c0 to %37 step %c1 iter_args(%arg9 = %49, %arg10 = %45, %arg11 = %41) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>) {
                %73 = arith.muli %arg8, %22 : index
                %74 = loom.subview %arg0[%29, 0, %73] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                loom.copy %74, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                %75 = loom.bufferize_to_tensor %63[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                %76 = linalg.fill ins(%cst_0 : f16) outs(%66 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                %77 = linalg.batch_matmul ins(%36, %75 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%76 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                loom.semaphore_give %63 : memref<?x128x?xf16>
                %78 = linalg.fill ins(%cst_2 : f16) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%77 : tensor<?x?x?xf16>) outs(%78 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.maximumf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<?x?xf16>
                %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %79 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.mulf %in_4, %cst_3 : f16
                  %93 = arith.cmpf ogt, %in, %92 : f16
                  %94 = arith.select %93, %in, %92 : f16
                  linalg.yield %94 : f16
                } -> tensor<?x?xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%77, %80 : tensor<?x?x?xf16>, tensor<?x?xf16>) outs(%66 : tensor<?x?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.mulf %in, %cst_3 : f16
                  %93 = arith.subf %92, %in_4 : f16
                  %94 = math.powf %cst, %93 : f16
                  linalg.yield %94 : f16
                } -> tensor<?x?x?xf16>
                %82 = linalg.fill ins(%cst_0 : f16) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%81 : tensor<?x?x?xf16>) outs(%82 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.addf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<?x?xf16>
                %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %80 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%61 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.subf %in, %in_4 : f16
                  %93 = math.powf %cst, %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %84, %83 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg10 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
                  %92 = arith.mulf %in, %in_4 : f16
                  %93 = arith.addf %92, %in_5 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?xf16>
                loom.semaphore_give %57 : memref<?x?xf16>
                %86 = loom.subview %arg1[%29, %73, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %86, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %87 = loom.bufferize_to_tensor %68[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %88 = linalg.fill ins(%cst_0 : f16) outs(%52 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %89 = linalg.batch_matmul ins(%81, %87 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%88 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %68 : memref<?x?x128xf16>
                loom.semaphore_give %65 : memref<?x?x?xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %arg11, %84 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?xf16>) outs(%arg11 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
                  %92 = arith.mulf %in_4, %in_5 : f16
                  %93 = arith.addf %in, %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %60 : memref<?x?xf16>
                loom.semaphore_give %51 : memref<?x?x128xf16>
                %91 = linalg.copy ins(%80 : tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %54 : memref<?x?xf16>
                scf.yield %91, %85, %90 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              loom.semaphore_give %47 : memref<?x?xf16>
              loom.semaphore_give %34 : memref<?x?x128xf16>
              %70 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%69#2, %69#1 : tensor<?x?x128xf16>, tensor<?x?xf16>) outs(%33 : tensor<?x?x128xf16>) {
              ^bb0(%in: f16, %in_4: f16, %out: f16):
                %73 = arith.divf %in, %in_4 : f16
                linalg.yield %73 : f16
              } -> tensor<?x?x128xf16>
              loom.semaphore_give %43 : memref<?x?xf16>
              loom.semaphore_give %39 : memref<?x?x128xf16>
              %71 = loom.subview %arg3[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              %72 = loom.bufferize_to_memref %70 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
              loom.copy %72, %71 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %32 : memref<?x?x128xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_flash__attention__x8_y8__d1i0_d0i1__f10__n_dim_x_level0_bc8_n_n(%arg0: memref<256x128x4096xf16>, %arg1: memref<256x4096x128xf16>, %arg2: memref<256x4096x128xf16>, %arg3: memref<256x4096x128xf16>) {
      %c8 = arith.constant 8 : index
      %cst = arith.constant 2.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.000000e+00 : f16
      %cst_2 = arith.constant 0xFC00 : f16
      %cst_3 = arith.constant 1.275630e-01 : f16
      %c256 = arith.constant 256 : index
      %c4096 = arith.constant 4096 : index
      %20 = loom.sym @tile_b {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c4096, %21 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          %25 = arith.ceildivui %24, %c8 : index
          scf.for %arg6 = %c0 to %25 step %c1 {
            %26 = arith.ceildivui %23, %c8 : index
            scf.for %arg7 = %c0 to %26 step %c1 {
              %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
              %28 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
              %29 = arith.muli %27, %20 : index
              %30 = arith.muli %28, %21 : index
              %31 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %32 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %33 = loom.init_tensor %32[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %34 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %35 = loom.subview %arg2[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
              %36 = loom.bufferize_to_tensor %34[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %37 = arith.ceildivui %c4096, %22 : index
              %38 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %39 = loom.semaphore_take %38 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %40 = loom.init_tensor %39[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
              %42 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
              %44 = loom.init_tensor %43[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %45 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %46 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
              %48 = loom.init_tensor %47[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %49 = linalg.fill ins(%cst_2 : f16) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %50 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %51 = loom.semaphore_take %50 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %52 = loom.init_tensor %51[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %53 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %54 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
              %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %56 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %57 = loom.semaphore_take %56 : memref<?x?xf16> -> memref<?x?xf16>
              %58 = loom.init_tensor %57[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %59 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %60 = loom.semaphore_take %59 : memref<?x?xf16> -> memref<?x?xf16>
              %61 = loom.init_tensor %60[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %62 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
              %63 = loom.semaphore_take %62 : memref<?x128x?xf16> -> memref<?x128x?xf16>
              %64 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
              %65 = loom.semaphore_take %64 : memref<?x?x?xf16> -> memref<?x?x?xf16>
              %66 = loom.init_tensor %65[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
              %67 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
              %68 = loom.semaphore_take %67 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %69:3 = scf.for %arg8 = %c0 to %37 step %c1 iter_args(%arg9 = %49, %arg10 = %45, %arg11 = %41) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>) {
                %73 = arith.muli %arg8, %22 : index
                %74 = loom.subview %arg0[%29, 0, %73] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                loom.copy %74, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                %75 = loom.bufferize_to_tensor %63[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                %76 = linalg.fill ins(%cst_0 : f16) outs(%66 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                %77 = linalg.batch_matmul ins(%36, %75 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%76 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                loom.semaphore_give %63 : memref<?x128x?xf16>
                %78 = linalg.fill ins(%cst_2 : f16) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%77 : tensor<?x?x?xf16>) outs(%78 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.maximumf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<?x?xf16>
                %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %79 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.mulf %in_4, %cst_3 : f16
                  %93 = arith.cmpf ogt, %in, %92 : f16
                  %94 = arith.select %93, %in, %92 : f16
                  linalg.yield %94 : f16
                } -> tensor<?x?xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%77, %80 : tensor<?x?x?xf16>, tensor<?x?xf16>) outs(%66 : tensor<?x?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.mulf %in, %cst_3 : f16
                  %93 = arith.subf %92, %in_4 : f16
                  %94 = math.powf %cst, %93 : f16
                  linalg.yield %94 : f16
                } -> tensor<?x?x?xf16>
                %82 = linalg.fill ins(%cst_0 : f16) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%81 : tensor<?x?x?xf16>) outs(%82 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.addf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<?x?xf16>
                %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %80 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%61 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.subf %in, %in_4 : f16
                  %93 = math.powf %cst, %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %84, %83 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg10 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
                  %92 = arith.mulf %in, %in_4 : f16
                  %93 = arith.addf %92, %in_5 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?xf16>
                loom.semaphore_give %57 : memref<?x?xf16>
                %86 = loom.subview %arg1[%29, %73, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %86, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %87 = loom.bufferize_to_tensor %68[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %88 = linalg.fill ins(%cst_0 : f16) outs(%52 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %89 = linalg.batch_matmul ins(%81, %87 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%88 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %68 : memref<?x?x128xf16>
                loom.semaphore_give %65 : memref<?x?x?xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %arg11, %84 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?xf16>) outs(%arg11 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
                  %92 = arith.mulf %in_4, %in_5 : f16
                  %93 = arith.addf %in, %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %60 : memref<?x?xf16>
                loom.semaphore_give %51 : memref<?x?x128xf16>
                %91 = linalg.copy ins(%80 : tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %54 : memref<?x?xf16>
                scf.yield %91, %85, %90 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              loom.semaphore_give %47 : memref<?x?xf16>
              loom.semaphore_give %34 : memref<?x?x128xf16>
              %70 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%69#2, %69#1 : tensor<?x?x128xf16>, tensor<?x?xf16>) outs(%33 : tensor<?x?x128xf16>) {
              ^bb0(%in: f16, %in_4: f16, %out: f16):
                %73 = arith.divf %in, %in_4 : f16
                linalg.yield %73 : f16
              } -> tensor<?x?x128xf16>
              loom.semaphore_give %43 : memref<?x?xf16>
              loom.semaphore_give %39 : memref<?x?x128xf16>
              %71 = loom.subview %arg3[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              %72 = loom.bufferize_to_memref %70 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
              loom.copy %72, %71 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %32 : memref<?x?x128xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_flash__attention__x8_y8__d1i0_d0i1__f10__n_dim_x_level0_bc8_dim_x_level0_bc8_n(%arg0: memref<256x128x4096xf16>, %arg1: memref<256x4096x128xf16>, %arg2: memref<256x4096x128xf16>, %arg3: memref<256x4096x128xf16>) {
      %c8 = arith.constant 8 : index
      %cst = arith.constant 2.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.000000e+00 : f16
      %cst_2 = arith.constant 0xFC00 : f16
      %cst_3 = arith.constant 1.275630e-01 : f16
      %c256 = arith.constant 256 : index
      %c4096 = arith.constant 4096 : index
      %20 = loom.sym @tile_b {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c4096, %21 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          %25 = arith.ceildivui %24, %c8 : index
          scf.for %arg6 = %c0 to %25 step %c1 {
            %26 = arith.ceildivui %23, %c8 : index
            scf.for %arg7 = %c0 to %26 step %c1 {
              %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
              %28 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
              %29 = arith.muli %27, %20 : index
              %30 = arith.muli %28, %21 : index
              %31 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %32 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %33 = loom.init_tensor %32[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %34 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %35 = loom.subview %arg2[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
              %36 = loom.bufferize_to_tensor %34[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %37 = arith.ceildivui %c4096, %22 : index
              %38 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %39 = loom.semaphore_take %38 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %40 = loom.init_tensor %39[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
              %42 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
              %44 = loom.init_tensor %43[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %45 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %46 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
              %48 = loom.init_tensor %47[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %49 = linalg.fill ins(%cst_2 : f16) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %50 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
              %51 = loom.semaphore_take %50 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %52 = loom.init_tensor %51[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
              %53 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %54 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
              %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %56 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %57 = loom.semaphore_take %56 : memref<?x?xf16> -> memref<?x?xf16>
              %58 = loom.init_tensor %57[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %59 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %60 = loom.semaphore_take %59 : memref<?x?xf16> -> memref<?x?xf16>
              %61 = loom.init_tensor %60[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %62 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
              %63 = loom.semaphore_take %62 : memref<?x128x?xf16> -> memref<?x128x?xf16>
              %64 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
              %65 = loom.semaphore_take %64 : memref<?x?x?xf16> -> memref<?x?x?xf16>
              %66 = loom.init_tensor %65[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
              %67 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
              %68 = loom.semaphore_take %67 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %69:3 = scf.for %arg8 = %c0 to %37 step %c1 iter_args(%arg9 = %49, %arg10 = %45, %arg11 = %41) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>) {
                %73 = arith.muli %arg8, %22 : index
                %74 = loom.subview %arg0[%29, 0, %73] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                loom.copy %74, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                %75 = loom.bufferize_to_tensor %63[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                %76 = linalg.fill ins(%cst_0 : f16) outs(%66 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                %77 = linalg.batch_matmul ins(%36, %75 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%76 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                loom.semaphore_give %63 : memref<?x128x?xf16>
                %78 = linalg.fill ins(%cst_2 : f16) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%77 : tensor<?x?x?xf16>) outs(%78 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.maximumf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<?x?xf16>
                %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %79 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.mulf %in_4, %cst_3 : f16
                  %93 = arith.cmpf ogt, %in, %92 : f16
                  %94 = arith.select %93, %in, %92 : f16
                  linalg.yield %94 : f16
                } -> tensor<?x?xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%77, %80 : tensor<?x?x?xf16>, tensor<?x?xf16>) outs(%66 : tensor<?x?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.mulf %in, %cst_3 : f16
                  %93 = arith.subf %92, %in_4 : f16
                  %94 = math.powf %cst, %93 : f16
                  linalg.yield %94 : f16
                } -> tensor<?x?x?xf16>
                %82 = linalg.fill ins(%cst_0 : f16) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%81 : tensor<?x?x?xf16>) outs(%82 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.addf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<?x?xf16>
                %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %80 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%61 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %92 = arith.subf %in, %in_4 : f16
                  %93 = math.powf %cst, %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %84, %83 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg10 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
                  %92 = arith.mulf %in, %in_4 : f16
                  %93 = arith.addf %92, %in_5 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?xf16>
                loom.semaphore_give %57 : memref<?x?xf16>
                %86 = loom.subview %arg1[%29, %73, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %86, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %87 = loom.bufferize_to_tensor %68[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %88 = linalg.fill ins(%cst_0 : f16) outs(%52 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %89 = linalg.batch_matmul ins(%81, %87 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%88 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %68 : memref<?x?x128xf16>
                loom.semaphore_give %65 : memref<?x?x?xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %arg11, %84 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?xf16>) outs(%arg11 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
                  %92 = arith.mulf %in_4, %in_5 : f16
                  %93 = arith.addf %in, %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %60 : memref<?x?xf16>
                loom.semaphore_give %51 : memref<?x?x128xf16>
                %91 = linalg.copy ins(%80 : tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %54 : memref<?x?xf16>
                scf.yield %91, %85, %90 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              loom.semaphore_give %47 : memref<?x?xf16>
              loom.semaphore_give %34 : memref<?x?x128xf16>
              %70 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%69#2, %69#1 : tensor<?x?x128xf16>, tensor<?x?xf16>) outs(%33 : tensor<?x?x128xf16>) {
              ^bb0(%in: f16, %in_4: f16, %out: f16):
                %73 = arith.divf %in, %in_4 : f16
                linalg.yield %73 : f16
              } -> tensor<?x?x128xf16>
              loom.semaphore_give %43 : memref<?x?xf16>
              loom.semaphore_give %39 : memref<?x?x128xf16>
              %71 = loom.subview %arg3[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              %72 = loom.bufferize_to_memref %70 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
              loom.copy %72, %71 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %32 : memref<?x?x128xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
}
