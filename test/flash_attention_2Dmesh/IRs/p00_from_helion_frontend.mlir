#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
module attributes {loom.tile_b = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
  func.func @_flash__attention(%arg0: memref<256x128x4096xf16>, %arg1: memref<256x4096x128xf16>, %arg2: memref<256x4096x128xf16>, %arg3: memref<256x4096x128xf16>) {
    %cst = arith.constant 2.000000e+00 : f16
    %c0_i64 = arith.constant 0 : i64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f16
    %cst_1 = arith.constant 1.000000e+00 : f16
    %cst_2 = arith.constant 0xFC00 : f16
    %cst_3 = arith.constant 1.275630e-01 : f16
    %c256 = arith.constant 256 : index
    %c4096 = arith.constant 4096 : index
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_b, upper_bound = 256 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_m, upper_bound = 4096 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 4096 : index} : () -> index
    %3 = arith.ceildivui %c256, %0 : index
    %4 = arith.ceildivui %c4096, %1 : index
    affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%3), symbol(%4)) {
      %5 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %6 = linalg.fill ins(%cst_2 : f16) outs(%5 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %7 = linalg.fill ins(%cst_1 : f16) outs(%5 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %8 = tensor.empty(%0, %1) : tensor<?x?x128xf16>
      %9 = linalg.fill ins(%cst_0 : f16) outs(%8 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
      %10 = arith.muli %arg4, %0 : index
      %11 = arith.muli %arg5, %1 : index
      %subview = memref.subview %arg2[%10, %11, 0] [%0, %1, 128] [1, 1, 1] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      %12 = bufferization.to_tensor %subview : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to tensor<?x?x128xf16>
      %13 = arith.ceildivui %c4096, %2 : index
      %14:3 = scf.for %arg6 = %c0 to %13 step %c1 iter_args(%arg7 = %6, %arg8 = %7, %arg9 = %9) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>) {
        %17 = arith.muli %arg6, %2 : index
        %subview_5 = memref.subview %arg0[%10, 0, %17] [%0, 128, %2] [1, 1, 1] : memref<256x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
        %18 = bufferization.to_tensor %subview_5 : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to tensor<?x128x?xf16>
        %19 = arith.index_cast %0 : index to i64
        %20 = arith.cmpi eq, %19, %19 : i64
        cf.assert %20, "mismatching contracting dimension"
        %21 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf16>
        %22 = linalg.fill ins(%cst_0 : f16) outs(%21 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %23 = linalg.batch_matmul ins(%12, %18 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%22 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %24 = tensor.empty(%0, %1) : tensor<?x?xi64>
        %25 = linalg.fill ins(%c0_i64 : i64) outs(%24 : tensor<?x?xi64>) -> tensor<?x?xi64>
        %26:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%23 : tensor<?x?x?xf16>) outs(%6, %25 : tensor<?x?xf16>, tensor<?x?xi64>) {
        ^bb0(%in: f16, %out: f16, %out_11: i64):
          %44 = linalg.index 2 : index
          %45 = arith.index_cast %44 : index to i64
          %46 = arith.maximumf %in, %out : f16
          %47 = arith.cmpf ogt, %in, %out : f16
          %48 = arith.select %47, %45, %out_11 : i64
          linalg.yield %46, %48 : f16, i64
        } -> (tensor<?x?xf16>, tensor<?x?xi64>)
        %27 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%26#0 : tensor<?x?xf16>) outs(%5 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %44 = arith.mulf %in, %cst_3 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?xf16>
        %28 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg7, %27 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%5 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %44 = arith.cmpf ogt, %in, %in_11 : f16
          %45 = arith.select %44, %in, %in_11 : f16
          linalg.yield %45 : f16
        } -> tensor<?x?xf16>
        %29 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23 : tensor<?x?x?xf16>) outs(%21 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %44 = arith.mulf %in, %cst_3 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?x?xf16>
        %extracted_slice_6 = tensor.extract_slice %28[0, 0] [%0, %1] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %expanded_7 = tensor.expand_shape %extracted_slice_6 [[0], [1, 2]] output_shape [%0, %1, 1] : tensor<?x?xf16> into tensor<?x?x1xf16>
        %30 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %expanded_7 : tensor<?x?x?xf16>, tensor<?x?x1xf16>) outs(%21 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %44 = arith.subf %in, %in_11 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?x?xf16>
        %31 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30 : tensor<?x?x?xf16>) outs(%21 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %44 = math.powf %cst, %in : f16
          linalg.yield %44 : f16
        } -> tensor<?x?x?xf16>
        %32 = linalg.fill ins(%cst_0 : f16) outs(%5 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %33 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%31 : tensor<?x?x?xf16>) outs(%32 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %44 = arith.addf %in, %out : f16
          linalg.yield %44 : f16
        } -> tensor<?x?xf16>
        %34 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg7, %28 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%5 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %44 = arith.subf %in, %in_11 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?xf16>
        %35 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%34 : tensor<?x?xf16>) outs(%5 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %44 = math.powf %cst, %in : f16
          linalg.yield %44 : f16
        } -> tensor<?x?xf16>
        %36 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg8, %35 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%5 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %44 = arith.mulf %in, %in_11 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?xf16>
        %37 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%36, %33 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%5 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %44 = arith.addf %in, %in_11 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?xf16>
        %extracted_slice_8 = tensor.extract_slice %35[0, 0] [%0, %1] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %expanded_9 = tensor.expand_shape %extracted_slice_8 [[0], [1, 2]] output_shape [%0, %1, 1] : tensor<?x?xf16> into tensor<?x?x1xf16>
        %38 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %expanded_9 : tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%8 : tensor<?x?x128xf16>) {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %44 = arith.mulf %in, %in_11 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?x128xf16>
        %subview_10 = memref.subview %arg1[%10, %17, 0] [%0, %2, 128] [1, 1, 1] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
        %39 = bufferization.to_tensor %subview_10 : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to tensor<?x?x128xf16>
        cf.assert %20, "mismatching contracting dimension"
        %40 = arith.index_cast %2 : index to i64
        %41 = arith.cmpi eq, %40, %40 : i64
        cf.assert %41, "mismatching contracting dimension"
        %42 = linalg.batch_matmul ins(%31, %39 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%9 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
        %43 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %38 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%8 : tensor<?x?x128xf16>) {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %44 = arith.addf %in, %in_11 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?x128xf16>
        scf.yield %28, %37, %43 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>
      }
      %extracted_slice = tensor.extract_slice %14#1[0, 0] [%0, %1] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
      %expanded = tensor.expand_shape %extracted_slice [[0], [1, 2]] output_shape [%0, %1, 1] : tensor<?x?xf16> into tensor<?x?x1xf16>
      %15 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14#2, %expanded : tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%8 : tensor<?x?x128xf16>) {
      ^bb0(%in: f16, %in_5: f16, %out: f16):
        %17 = arith.divf %in, %in_5 : f16
        linalg.yield %17 : f16
      } -> tensor<?x?x128xf16>
      %subview_4 = memref.subview %arg3[%10, %11, 0] [%0, %1, 128] [1, 1, 1] : memref<256x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      %16 = bufferization.to_buffer %15 : tensor<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      memref.copy %16, %subview_4 : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
    }
    return
  }
}

