#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 128 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 128 : index}} {
  func.func @_mamba_chunk_scan(%arg0: memref<2x8x8x512x512xf16>, %arg1: memref<2x128x8x512xf16>, %arg2: memref<2x128x8x512xf16>, %arg3: memref<2x4096x128x128xf16>, %arg4: memref<2x4096x8x128xf16>, %arg5: memref<2x8x128x128x128xf16>, %arg6: memref<128xf16>, %arg7: memref<2x4096x128x128xf16>) {
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c512 = arith.constant 512 : index
    %c128 = arith.constant 128 : index
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_m, upper_bound = 512 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 128 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_k, upper_bound = 8192 : index} : () -> index
    %3 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_h, upper_bound = 128 : index} : () -> index
    %4 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_b, upper_bound = 2 : index} : () -> index
    %5 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_c, upper_bound = 8 : index} : () -> index
    %6 = arith.ceildivui %c128, %3 : index
    %7 = arith.ceildivui %c512, %0 : index
    %8 = arith.ceildivui %c128, %1 : index
    %9 = arith.ceildivui %c2, %4 : index
    %10 = arith.ceildivui %c8, %5 : index
    affine.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (0, 0, 0, 0, 0) to (symbol(%6), symbol(%7), symbol(%8), symbol(%9), symbol(%10)) {
      %11 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %12 = linalg.fill ins(%cst : f16) outs(%11 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %13 = arith.muli %arg11, %4 : index
      %14 = arith.muli %arg8, %3 : index
      %15 = arith.muli %arg12, %5 : index
      %16 = arith.muli %arg9, %0 : index
      %subview = memref.subview %arg1[%13, %14, %15, %16] [1, 1, 1, %0] [1, 1, 1, 1] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
      %17 = bufferization.to_tensor %subview : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
      %18 = tensor.empty(%0) : tensor<?x32xf16>
      %19 = "loom.broadcast"(%17, %18) {dim = 1 : i64} : (tensor<?xf16>, tensor<?x32xf16>) -> tensor<?x?xf16>
      %20 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%19 : tensor<?x?xf16>) outs(%11 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %out: f16):
        %38 = math.exp %in : f16
        linalg.yield %38 : f16
      } -> tensor<?x?xf16>
      %21 = arith.muli %15, %c512 : index
      %22 = arith.addi %16, %21 : index
      %23 = arith.divui %14, %c16 : index
      %subview_0 = memref.subview %arg4[%13, %22, %23, 0] [1, %0, 1, 128] [1, 1, 1, 1] : memref<2x4096x8x128xf16> to memref<?x128xf16, strided<[1024, 1], offset: ?>>
      %24 = bufferization.to_tensor %subview_0 : memref<?x128xf16, strided<[1024, 1], offset: ?>> to tensor<?x128xf16>
      %25 = arith.muli %arg10, %1 : index
      %subview_1 = memref.subview %arg5[%13, %15, %14, 0, %25] [1, 1, 1, 128, %1] [1, 1, 1, 1, 1] : memref<2x8x128x128x128xf16> to memref<128x?xf16, strided<[128, 1], offset: ?>>
      %26 = bufferization.to_tensor %subview_1 : memref<128x?xf16, strided<[128, 1], offset: ?>> to tensor<128x?xf16>
      %27 = linalg.matmul ins(%24, %26 : tensor<?x128xf16>, tensor<128x?xf16>) outs(%12 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %28 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%27, %20 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%11 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_5: f16, %out: f16):
        %38 = arith.mulf %in, %in_5 : f16
        linalg.yield %38 : f16
      } -> tensor<?x?xf16>
      %29 = arith.addi %arg9, %c1 : index
      %30 = arith.muli %29, %0 : index
      %31 = arith.ceildivui %30, %2 : index
      %32 = scf.for %arg13 = %c0 to %31 step %c1 iter_args(%arg14 = %28) -> (tensor<?x?xf16>) {
        %38 = arith.muli %arg13, %2 : index
        %39 = arith.addi %38, %2 : index
        %40 = arith.cmpi ult, %39, %30 : index
        %41 = arith.select %40, %39, %30 : index
        %42 = arith.subi %41, %38 : index
        %subview_5 = memref.subview %arg0[%13, %15, %23, %16, %38] [1, 1, 1, %0, %42] [1, 1, 1, 1, 1] : memref<2x8x8x512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
        %43 = bufferization.to_tensor %subview_5 : memref<?x?xf16, strided<[512, 1], offset: ?>> to tensor<?x?xf16>
        %subview_6 = memref.subview %arg1[%13, %14, %15, %38] [1, 1, 1, %42] [1, 1, 1, 1] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
        %44 = bufferization.to_tensor %subview_6 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %45 = "loom.broadcast"(%17, %18) {dim = 1 : i64} : (tensor<?xf16>, tensor<?x32xf16>) -> tensor<?x?xf16>
        %46 = tensor.empty(%2) : tensor<32x?xf16>
        %47 = "loom.broadcast"(%44, %46) {dim = 0 : i64} : (tensor<?xf16>, tensor<32x?xf16>) -> tensor<?x?xf16>
        %48 = tensor.empty(%0, %2) : tensor<?x?xf16>
        %49 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%45, %47 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%48 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_9: f16, %out: f16):
          %58 = arith.subf %in, %in_9 : f16
          linalg.yield %58 : f16
        } -> tensor<?x?xf16>
        %50 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%49 : tensor<?x?xf16>) outs(%48 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %58 = math.exp %in : f16
          linalg.yield %58 : f16
        } -> tensor<?x?xf16>
        %51 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%43, %50 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%48 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_9: f16, %out: f16):
          %58 = arith.mulf %in, %in_9 : f16
          linalg.yield %58 : f16
        } -> tensor<?x?xf16>
        %subview_7 = memref.subview %arg2[%13, %14, %15, %38] [1, 1, 1, %42] [1, 1, 1, 1] : memref<2x128x8x512xf16> to memref<?xf16, strided<[1], offset: ?>>
        %52 = bufferization.to_tensor %subview_7 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %53 = "loom.broadcast"(%52, %46) {dim = 0 : i64} : (tensor<?xf16>, tensor<32x?xf16>) -> tensor<?x?xf16>
        %54 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%51, %53 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%48 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_9: f16, %out: f16):
          %58 = arith.mulf %in, %in_9 : f16
          linalg.yield %58 : f16
        } -> tensor<?x?xf16>
        %55 = arith.addi %38, %21 : index
        %subview_8 = memref.subview %arg3[%13, %55, %14, %25] [1, %2, 1, %1] [1, 1, 1, 1] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
        %56 = bufferization.to_tensor %subview_8 : memref<?x?xf16, strided<[16384, 1], offset: ?>> to tensor<?x?xf16>
        %57 = linalg.matmul ins(%54, %56 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg14 : tensor<?x?xf16>) -> tensor<?x?xf16>
        scf.yield %57 : tensor<?x?xf16>
      }
      %subview_2 = memref.subview %arg6[%14] [1] [1] : memref<128xf16> to memref<f16, strided<[], offset: ?>>
      %33 = bufferization.to_tensor %subview_2 : memref<f16, strided<[], offset: ?>> to tensor<f16>
      %subview_3 = memref.subview %arg3[%13, %22, %14, %25] [1, %0, 1, %1] [1, 1, 1, 1] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
      %34 = bufferization.to_tensor %subview_3 : memref<?x?xf16, strided<[16384, 1], offset: ?>> to tensor<?x?xf16>
      %35 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%34, %33 : tensor<?x?xf16>, tensor<f16>) outs(%11 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_5: f16, %out: f16):
        %38 = arith.mulf %in, %in_5 : f16
        linalg.yield %38 : f16
      } -> tensor<?x?xf16>
      %36 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%32, %35 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%11 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_5: f16, %out: f16):
        %38 = arith.addf %in, %in_5 : f16
        linalg.yield %38 : f16
      } -> tensor<?x?xf16>
      %subview_4 = memref.subview %arg7[%13, %22, %14, %25] [1, %0, 1, %1] [1, 1, 1, 1] : memref<2x4096x128x128xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
      %37 = bufferization.to_buffer %36 : tensor<?x?xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
      memref.copy %37, %subview_4 : memref<?x?xf16, strided<[16384, 1], offset: ?>> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
    }
    return
  }
}

