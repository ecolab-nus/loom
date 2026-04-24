module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 4096 : index}} {
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
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_mqa_decode__x8_y8__d0i1_d1i0__f01__dim_x_level0_bc8_n_n_n__is_double_buffer1__tile_b1__tile_n512__tile_s512(%arg0: memref<8x128x4096xf16>, %arg1: memref<8x4096x128xf16>, %arg2: memref<8x64x128xf16>, %arg3: memref<8x64x128xf16>) {
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c8192 = arith.constant 8192 : index
      %c7 = arith.constant 7 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c512 = arith.constant 512 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %20 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
        %21 = loom.semaphore_take %20 : memref<1x64x128xf16> -> memref<1x64x128xf16>
        %22 = loom.semaphore_take %20 : memref<1x64x128xf16> -> memref<1x64x128xf16>
        %23 = loom.semaphore_take %20 : memref<1x64x128xf16> -> memref<1x64x128xf16>
        %24 = arith.muli %arg4, %c8192 overflow<nsw> : index
        %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [1, 64, 128], strides: [8192, 128, 1] : memref<8x64x128xf16> to memref<1x64x128xf16, strided<[8192, 128, 1], offset: ?>>
        loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<1x64x128xf16, strided<[8192, 128, 1], offset: ?>> to memref<1x64x128xf16>
        %25 = arith.muli %arg5, %c512 : index
        %26 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
        %27 = loom.semaphore_take %26 : memref<1x64x128xf16> -> memref<1x64x128xf16>
        %28 = loom.semaphore_take %26 : memref<1x64x128xf16> -> memref<1x64x128xf16>
        %29 = loom.semaphore_take %26 : memref<1x64x128xf16> -> memref<1x64x128xf16>
        linalg.fill ins(%cst : f16) outs(%29 : memref<1x64x128xf16>)
        %30 = loom.alloc [1, 64, 1] on @L1 : memref<1x64x1xf16>
        %31 = loom.semaphore_take %30 : memref<1x64x1xf16> -> memref<1x64x1xf16>
        %32 = loom.semaphore_take %30 : memref<1x64x1xf16> -> memref<1x64x1xf16>
        %33 = loom.semaphore_take %30 : memref<1x64x1xf16> -> memref<1x64x1xf16>
        %34 = loom.semaphore_take %30 : memref<1x64x1xf16> -> memref<1x64x1xf16>
        linalg.fill ins(%cst_0 : f16) outs(%34 : memref<1x64x1xf16>)
        %35 = loom.alloc [1, 64, 1] on @L1 : memref<1x64x1xf16>
        %36 = loom.semaphore_take %35 : memref<1x64x1xf16> -> memref<1x64x1xf16>
        %37 = loom.semaphore_take %35 : memref<1x64x1xf16> -> memref<1x64x1xf16>
        linalg.fill ins(%cst_1 : f16) outs(%37 : memref<1x64x1xf16>)
        %38 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
        %39 = loom.semaphore_take %38 : memref<1x64x128xf16> -> memref<1x64x128xf16>
        %40 = loom.alloc [1, 64, 1] on @L1 : memref<1x64x1xf16>
        %41 = loom.semaphore_take %40 : memref<1x64x1xf16> -> memref<1x64x1xf16>
        %42 = loom.alloc [1, 64, 1] on @L1 : memref<1x64x1xf16>
        %43 = loom.semaphore_take %42 : memref<1x64x1xf16> -> memref<1x64x1xf16>
        %44 = loom.alloc [1, 64, 1] on @L1 : memref<1x64x1xf16>
        %45 = loom.semaphore_take %44 : memref<1x64x1xf16> -> memref<1x64x1xf16>
        %46 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
        %47 = loom.semaphore_take %46 : memref<1x128x512xf16> -> memref<1x128x512xf16>
        %48 = loom.alloc [1, 64, 512] on @L1 : memref<1x64x512xf16>
        %49 = loom.semaphore_take %48 : memref<1x64x512xf16> -> memref<1x64x512xf16>
        %50 = loom.alloc [1, 64, 32] on @L1 : memref<1x64x32xf16>
        %51 = loom.semaphore_take %50 : memref<1x64x32xf16> -> memref<1x64x32xf16>
        %52 = loom.semaphore_take %50 : memref<1x64x32xf16> -> memref<1x64x32xf16>
        %53 = loom.semaphore_take %50 : memref<1x64x32xf16> -> memref<1x64x32xf16>
        %54 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
        %55 = loom.semaphore_take %54 : memref<1x512x128xf16> -> memref<1x512x128xf16>
        %56 = arith.muli %arg4, %c524288 overflow<nsw> : index
        %57 = arith.addi %56, %25 : index
        %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%57], sizes: [1, 128, 512], strides: [524288, 4096, 1] : memref<8x128x4096xf16> to memref<1x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_3, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x512xf16>
        loom.batch_matmul ins(%23, %47 : memref<1x64x128xf16>, memref<1x128x512xf16>) outs(%49 : memref<1x64x512xf16>)
        loom.semaphore_give %47 : memref<1x128x512xf16>
        linalg.fill ins(%cst_1 : f16) outs(%41 : memref<1x64x1xf16>)
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%49 : memref<1x64x512xf16>) outs(%41 : memref<1x64x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %74 = arith.maximumf %in, %out : f16
          linalg.yield %74 : f16
        }
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %41 : memref<1x64x1xf16>, memref<1x64x1xf16>) outs(%41 : memref<1x64x1xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %74 = arith.mulf %in_5, %cst_2 : f16
          %75 = arith.cmpf ogt, %in, %74 : f16
          %76 = arith.select %75, %in, %74 : f16
          linalg.yield %76 : f16
        }
        %58 = loom.broadcast ins(%41 : memref<1x64x1xf16>) outs(%53 : memref<1x64x32xf16>) dim(2) -> memref<1x64x512xf16, strided<[?, ?, ?], offset: ?>>
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%49, %58 : memref<1x64x512xf16>, memref<1x64x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%49 : memref<1x64x512xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %74 = arith.mulf %in, %cst_2 : f16
          %75 = arith.subf %74, %in_5 : f16
          %76 = math.exp %75 : f16
          linalg.yield %76 : f16
        }
        loom.semaphore_give %53 : memref<1x64x32xf16>
        linalg.fill ins(%cst : f16) outs(%43 : memref<1x64x1xf16>)
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%49 : memref<1x64x512xf16>) outs(%43 : memref<1x64x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %74 = arith.addf %in, %out : f16
          linalg.yield %74 : f16
        }
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %41 : memref<1x64x1xf16>, memref<1x64x1xf16>) outs(%45 : memref<1x64x1xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %74 = arith.subf %in, %in_5 : f16
          %75 = math.exp %74 : f16
          linalg.yield %75 : f16
        }
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %45, %43 : memref<1x64x1xf16>, memref<1x64x1xf16>, memref<1x64x1xf16>) outs(%34 : memref<1x64x1xf16>) {
        ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
          %74 = arith.mulf %in, %in_5 : f16
          %75 = arith.addf %74, %in_6 : f16
          linalg.yield %75 : f16
        }
        loom.semaphore_give %43 : memref<1x64x1xf16>
        %59 = loom.broadcast ins(%45 : memref<1x64x1xf16>) outs(%52 : memref<1x64x32xf16>) dim(2) -> memref<1x64x128xf16, strided<[?, ?, ?], offset: ?>>
        loom.semaphore_give %45 : memref<1x64x1xf16>
        %60 = arith.muli %arg5, %c65536 : index
        %61 = arith.addi %56, %60 : index
        %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%61], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<8x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
        loom.copy %reinterpret_cast_4, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x512x128xf16>
        loom.batch_matmul ins(%49, %55 : memref<1x64x512xf16>, memref<1x512x128xf16>) outs(%39 : memref<1x64x128xf16>)
        loom.semaphore_give %55 : memref<1x512x128xf16>
        loom.semaphore_give %49 : memref<1x64x512xf16>
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %29, %59 : memref<1x64x128xf16>, memref<1x64x128xf16>, memref<1x64x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%29 : memref<1x64x128xf16>) {
        ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
          %74 = arith.mulf %in_5, %in_6 : f16
          %75 = arith.addf %in, %74 : f16
          linalg.yield %75 : f16
        }
        loom.semaphore_give %52 : memref<1x64x32xf16>
        loom.semaphore_give %39 : memref<1x64x128xf16>
        linalg.copy ins(%41 : memref<1x64x1xf16>) outs(%37 : memref<1x64x1xf16>)
        loom.semaphore_give %41 : memref<1x64x1xf16>
        loom.semaphore_give %23 : memref<1x64x128xf16>
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %37 : memref<1x64x1xf16>, memref<1x64x1xf16>) outs(%36 : memref<1x64x1xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %74 = math.log %in : f16
          %75 = arith.addf %74, %in_5 : f16
          linalg.yield %75 : f16
        }
        loom.semaphore_give %37 : memref<1x64x1xf16>
        %62 = loom.broadcast ins(%34 : memref<1x64x1xf16>) outs(%51 : memref<1x64x32xf16>) dim(2) -> memref<1x64x128xf16, strided<[?, ?, ?], offset: ?>>
        loom.semaphore_give %34 : memref<1x64x1xf16>
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %62 : memref<1x64x128xf16>, memref<1x64x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%22 : memref<1x64x128xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %74 = arith.divf %in, %in_5 : f16
          linalg.yield %74 : f16
        }
        loom.semaphore_give %51 : memref<1x64x32xf16>
        loom.semaphore_give %29 : memref<1x64x128xf16>
        loom.sync ins(%36 : memref<1x64x1xf16>) outs(%33 : memref<1x64x1xf16>)
        loom.semaphore_give %36 : memref<1x64x1xf16>
        %63 = loom.alloc [8, 1, 64, 1] on @L1 : memref<8x1x64x1xf16>
        %64 = loom.semaphore_take %63 : memref<8x1x64x1xf16> -> memref<8x1x64x1xf16>
        loom.gather ins(%33 : memref<1x64x1xf16>) outs(%64 : memref<8x1x64x1xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4])
        loom.semaphore_give %33 : memref<1x64x1xf16>
        loom.sync ins(%22 : memref<1x64x128xf16>) outs(%28 : memref<1x64x128xf16>)
        loom.semaphore_give %22 : memref<1x64x128xf16>
        %65 = loom.alloc [8, 1, 64, 128] on @L1 : memref<8x1x64x128xf16>
        %66 = loom.semaphore_take %65 : memref<8x1x64x128xf16> -> memref<8x1x64x128xf16>
        loom.gather ins(%28 : memref<1x64x128xf16>) outs(%66 : memref<8x1x64x128xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4])
        loom.semaphore_give %28 : memref<1x64x128xf16>
        %67 = loom.alloc [8, 1, 64, 1] on @L1 : memref<8x1x64x1xf16>
        %68 = loom.semaphore_take %67 : memref<8x1x64x1xf16> -> memref<8x1x64x1xf16>
        %69 = loom.alloc [8, 1, 64, 128] on @L1 : memref<8x1x64x128xf16>
        %70 = loom.semaphore_take %69 : memref<8x1x64x128xf16> -> memref<8x1x64x128xf16>
        %71 = loom.alloc [8, 1, 64, 32] on @L1 : memref<8x1x64x32xf16>
        %72 = loom.semaphore_take %71 : memref<8x1x64x32xf16> -> memref<8x1x64x32xf16>
        %73 = arith.cmpi eq, %arg5, %c0 : index
        scf.if %73 {
          linalg.fill ins(%cst_1 : f16) outs(%32 : memref<1x64x1xf16>)
          loom.sync ins(%64 : memref<8x1x64x1xf16>) outs(%68 : memref<8x1x64x1xf16>)
          loom.semaphore_give %64 : memref<8x1x64x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%68 : memref<8x1x64x1xf16>) outs(%32 : memref<1x64x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %75 = arith.maximumf %in, %out : f16
            linalg.yield %75 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%68, %32 : memref<8x1x64x1xf16>, memref<1x64x1xf16>) outs(%68 : memref<8x1x64x1xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %75 = arith.subf %in, %in_6 : f16
            %76 = math.exp %75 : f16
            linalg.yield %76 : f16
          }
          loom.semaphore_give %32 : memref<1x64x1xf16>
          linalg.fill ins(%cst : f16) outs(%31 : memref<1x64x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%68 : memref<8x1x64x1xf16>) outs(%31 : memref<1x64x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %75 = arith.addf %in, %out : f16
            linalg.yield %75 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%68, %31 : memref<8x1x64x1xf16>, memref<1x64x1xf16>) outs(%68 : memref<8x1x64x1xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %75 = arith.divf %in, %in_6 : f16
            linalg.yield %75 : f16
          }
          loom.semaphore_give %31 : memref<1x64x1xf16>
          %74 = loom.broadcast ins(%68 : memref<8x1x64x1xf16>) outs(%72 : memref<8x1x64x32xf16>) dim(3) -> memref<8x1x64x128xf16, strided<[?, ?, ?, ?], offset: ?>>
          loom.semaphore_give %68 : memref<8x1x64x1xf16>
          loom.sync ins(%66 : memref<8x1x64x128xf16>) outs(%70 : memref<8x1x64x128xf16>)
          loom.semaphore_give %66 : memref<8x1x64x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%70, %74 : memref<8x1x64x128xf16>, memref<8x1x64x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%70 : memref<8x1x64x128xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %75 = arith.mulf %in, %in_6 : f16
            linalg.yield %75 : f16
          }
          loom.semaphore_give %72 : memref<8x1x64x32xf16>
          linalg.fill ins(%cst : f16) outs(%21 : memref<1x64x128xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%70 : memref<8x1x64x128xf16>) outs(%21 : memref<1x64x128xf16>) {
          ^bb0(%in: f16, %out: f16):
            %75 = arith.addf %in, %out : f16
            linalg.yield %75 : f16
          }
          loom.semaphore_give %70 : memref<8x1x64x128xf16>
          loom.sync ins(%21 : memref<1x64x128xf16>) outs(%27 : memref<1x64x128xf16>)
          loom.semaphore_give %21 : memref<1x64x128xf16>
          %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64, 128], strides: [8192, 128, 1] : memref<8x64x128xf16> to memref<1x64x128xf16, strided<[8192, 128, 1], offset: ?>>
          loom.copy %27, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x64x128xf16> to memref<1x64x128xf16, strided<[8192, 128, 1], offset: ?>>
          loom.semaphore_give %27 : memref<1x64x128xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
}
