module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
  %0 = df.mat "FPU" {shape = [32, 32, 32], throughput = 128}
  %1 = df.vec "SFPU" {shape = [32]}
  %2 = df.spatial_dim "x", 8
  %3 = df.spatial_dim "y", 8
  %4 = df.core "core" {scaleout=(%2, %3) , scalein=(%0, %1, [8, 1])}
  %5 = df.memory "L1" {scaleout=(%2, %3) , size = 1499136, bandwidth = 15}
  %6 = df.mux %4 : !df.compute, %5 : !df.memory  {map = affine_map<(d0, d1) -> (d0, d1)>}
  %7 = df.interconnects "horizontal_links" %5 : !df.memory, %5 : !df.memory  {bandwidth = 128 : i64, map = affine_map<(d0, d1) -> ((d0 + 1) mod 8, d1)>, spatial_dims = [@x]} : !df.interconnect
  %8 = df.interconnects "vertical_links" %5 : !df.memory, %5 : !df.memory  {bandwidth = 128 : i64, map = affine_map<(d0, d1) -> (d0, (d1 + 1) mod 8)>, spatial_dims = [@y]} : !df.interconnect
  %9 = df.spatial_dim "d", 4
  %10 = df.memory "DRAM" {scaleout=(%9) , size = 34359738368, bandwidth = 288}
  %11 = df.interconnects "NoC" %5 : !df.memory, %10 : !df.memory  {map = affine_map<(d0, d1) -> (d0 ceildiv 4 + (d1 ceildiv 4) * 2)>} : !df.interconnect
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f01__d_d__block_size_064__block_size_14096__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 4096] on @L1 : memref<512x4096xf16>
        %15 = loom.semaphore_take %14 : memref<512x4096xf16> -> memref<512x4096xf16>
        %16 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %17 = loom.semaphore_take %16 : memref<64x512xf16> -> memref<64x512xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        %20 = arith.muli %13, %c32768 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x512xf16, strided<[512, 1], offset: ?>>, memref<64x512xf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<512x4096xf16, strided<[4096, 1]>>
        %cast = memref.cast %reinterpret_cast_0 : memref<512x4096xf16, strided<[4096, 1]>> to memref<512x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x4096xf16, strided<[4096, 1], offset: ?>>, memref<512x4096xf16>
        linalg.matmul ins(%17, %15 : memref<64x512xf16>, memref<512x4096xf16>) outs(%19 : memref<64x4096xf16>)
        loom.semaphore_give %15 : memref<512x4096xf16>
        loom.semaphore_give %17 : memref<64x512xf16>
        %21 = arith.muli %13, %c262144 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f01__d_a__block_size_064__block_size_14096__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 4096] on @L1 : memref<512x4096xf16>
        %15 = loom.semaphore_take %14 : memref<512x4096xf16> -> memref<512x4096xf16>
        %16 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %17 = loom.semaphore_take %16 : memref<64x512xf16> -> memref<64x512xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        %20 = arith.muli %13, %c32768 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x512xf16, strided<[512, 1], offset: ?>>, memref<64x512xf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<512x4096xf16, strided<[4096, 1]>>
        %cast = memref.cast %reinterpret_cast_0 : memref<512x4096xf16, strided<[4096, 1]>> to memref<512x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<512x4096xf16, strided<[4096, 1], offset: ?>>, memref<512x4096xf16>
        linalg.matmul ins(%17, %15 : memref<64x512xf16>, memref<512x4096xf16>) outs(%19 : memref<64x4096xf16>)
        loom.semaphore_give %15 : memref<512x4096xf16>
        loom.semaphore_give %17 : memref<64x512xf16>
        %21 = arith.muli %13, %c262144 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f01__d_h__block_size_064__block_size_14096__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 4096] on @L1 : memref<512x4096xf16>
        %15 = loom.semaphore_take %14 : memref<512x4096xf16> -> memref<512x4096xf16>
        %16 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %17 = loom.semaphore_take %16 : memref<64x512xf16> -> memref<64x512xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        %20 = arith.muli %13, %c32768 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x512xf16, strided<[512, 1], offset: ?>>, memref<64x512xf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<512x4096xf16, strided<[4096, 1]>>
        %cast = memref.cast %reinterpret_cast_0 : memref<512x4096xf16, strided<[4096, 1]>> to memref<512x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<512x4096xf16, strided<[4096, 1], offset: ?>>, memref<512x4096xf16>
        linalg.matmul ins(%17, %15 : memref<64x512xf16>, memref<512x4096xf16>) outs(%19 : memref<64x4096xf16>)
        loom.semaphore_give %15 : memref<512x4096xf16>
        loom.semaphore_give %17 : memref<64x512xf16>
        %21 = arith.muli %13, %c262144 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f01__d_v__block_size_064__block_size_14096__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 4096] on @L1 : memref<512x4096xf16>
        %15 = loom.semaphore_take %14 : memref<512x4096xf16> -> memref<512x4096xf16>
        %16 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %17 = loom.semaphore_take %16 : memref<64x512xf16> -> memref<64x512xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        %20 = arith.muli %13, %c32768 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x512xf16, strided<[512, 1], offset: ?>>, memref<64x512xf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<512x4096xf16, strided<[4096, 1]>>
        %cast = memref.cast %reinterpret_cast_0 : memref<512x4096xf16, strided<[4096, 1]>> to memref<512x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<512x4096xf16, strided<[4096, 1], offset: ?>>, memref<512x4096xf16>
        linalg.matmul ins(%17, %15 : memref<64x512xf16>, memref<512x4096xf16>) outs(%19 : memref<64x4096xf16>)
        loom.semaphore_give %15 : memref<512x4096xf16>
        loom.semaphore_give %17 : memref<64x512xf16>
        %21 = arith.muli %13, %c262144 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f10__d_d__block_size_064__block_size_14096__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 4096] on @L1 : memref<512x4096xf16>
        %15 = loom.semaphore_take %14 : memref<512x4096xf16> -> memref<512x4096xf16>
        %16 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %17 = loom.semaphore_take %16 : memref<64x512xf16> -> memref<64x512xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        %20 = arith.muli %13, %c32768 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x512xf16, strided<[512, 1], offset: ?>>, memref<64x512xf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<512x4096xf16, strided<[4096, 1]>>
        %cast = memref.cast %reinterpret_cast_0 : memref<512x4096xf16, strided<[4096, 1]>> to memref<512x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x4096xf16, strided<[4096, 1], offset: ?>>, memref<512x4096xf16>
        linalg.matmul ins(%17, %15 : memref<64x512xf16>, memref<512x4096xf16>) outs(%19 : memref<64x4096xf16>)
        loom.semaphore_give %15 : memref<512x4096xf16>
        loom.semaphore_give %17 : memref<64x512xf16>
        %21 = arith.muli %13, %c262144 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f10__d_a__block_size_064__block_size_14096__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 4096] on @L1 : memref<512x4096xf16>
        %15 = loom.semaphore_take %14 : memref<512x4096xf16> -> memref<512x4096xf16>
        %16 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %17 = loom.semaphore_take %16 : memref<64x512xf16> -> memref<64x512xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        %20 = arith.muli %13, %c32768 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x512xf16, strided<[512, 1], offset: ?>>, memref<64x512xf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<512x4096xf16, strided<[4096, 1]>>
        %cast = memref.cast %reinterpret_cast_0 : memref<512x4096xf16, strided<[4096, 1]>> to memref<512x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<512x4096xf16, strided<[4096, 1], offset: ?>>, memref<512x4096xf16>
        linalg.matmul ins(%17, %15 : memref<64x512xf16>, memref<512x4096xf16>) outs(%19 : memref<64x4096xf16>)
        loom.semaphore_give %15 : memref<512x4096xf16>
        loom.semaphore_give %17 : memref<64x512xf16>
        %21 = arith.muli %13, %c262144 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f01__d_d__block_size_064__block_size_14096__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 4096] on @L1 : memref<512x4096xf16>
        %15 = loom.semaphore_take %14 : memref<512x4096xf16> -> memref<512x4096xf16>
        %16 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %17 = loom.semaphore_take %16 : memref<64x512xf16> -> memref<64x512xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        %20 = arith.muli %13, %c32768 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x512xf16, strided<[512, 1], offset: ?>>, memref<64x512xf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<512x4096xf16, strided<[4096, 1]>>
        %cast = memref.cast %reinterpret_cast_0 : memref<512x4096xf16, strided<[4096, 1]>> to memref<512x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x4096xf16, strided<[4096, 1], offset: ?>>, memref<512x4096xf16>
        linalg.matmul ins(%17, %15 : memref<64x512xf16>, memref<512x4096xf16>) outs(%19 : memref<64x4096xf16>)
        loom.semaphore_give %15 : memref<512x4096xf16>
        loom.semaphore_give %17 : memref<64x512xf16>
        %21 = arith.muli %13, %c262144 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f01__d_a__block_size_064__block_size_14096__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 4096] on @L1 : memref<512x4096xf16>
        %15 = loom.semaphore_take %14 : memref<512x4096xf16> -> memref<512x4096xf16>
        %16 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %17 = loom.semaphore_take %16 : memref<64x512xf16> -> memref<64x512xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        %20 = arith.muli %13, %c32768 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x512xf16, strided<[512, 1], offset: ?>>, memref<64x512xf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<512x4096xf16, strided<[4096, 1]>>
        %cast = memref.cast %reinterpret_cast_0 : memref<512x4096xf16, strided<[4096, 1]>> to memref<512x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<512x4096xf16, strided<[4096, 1], offset: ?>>, memref<512x4096xf16>
        linalg.matmul ins(%17, %15 : memref<64x512xf16>, memref<512x4096xf16>) outs(%19 : memref<64x4096xf16>)
        loom.semaphore_give %15 : memref<512x4096xf16>
        loom.semaphore_give %17 : memref<64x512xf16>
        %21 = arith.muli %13, %c262144 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f01__d_h__block_size_064__block_size_14096__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 4096] on @L1 : memref<512x4096xf16>
        %15 = loom.semaphore_take %14 : memref<512x4096xf16> -> memref<512x4096xf16>
        %16 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %17 = loom.semaphore_take %16 : memref<64x512xf16> -> memref<64x512xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        %20 = arith.muli %13, %c32768 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x512xf16, strided<[512, 1], offset: ?>>, memref<64x512xf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<512x4096xf16, strided<[4096, 1]>>
        %cast = memref.cast %reinterpret_cast_0 : memref<512x4096xf16, strided<[4096, 1]>> to memref<512x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<512x4096xf16, strided<[4096, 1], offset: ?>>, memref<512x4096xf16>
        linalg.matmul ins(%17, %15 : memref<64x512xf16>, memref<512x4096xf16>) outs(%19 : memref<64x4096xf16>)
        loom.semaphore_give %15 : memref<512x4096xf16>
        loom.semaphore_give %17 : memref<64x512xf16>
        %21 = arith.muli %13, %c262144 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f01__d_v__block_size_064__block_size_14096__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 4096] on @L1 : memref<512x4096xf16>
        %15 = loom.semaphore_take %14 : memref<512x4096xf16> -> memref<512x4096xf16>
        %16 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %17 = loom.semaphore_take %16 : memref<64x512xf16> -> memref<64x512xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        %20 = arith.muli %13, %c32768 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x512xf16, strided<[512, 1], offset: ?>>, memref<64x512xf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<512x4096xf16, strided<[4096, 1]>>
        %cast = memref.cast %reinterpret_cast_0 : memref<512x4096xf16, strided<[4096, 1]>> to memref<512x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<512x4096xf16, strided<[4096, 1], offset: ?>>, memref<512x4096xf16>
        linalg.matmul ins(%17, %15 : memref<64x512xf16>, memref<512x4096xf16>) outs(%19 : memref<64x4096xf16>)
        loom.semaphore_give %15 : memref<512x4096xf16>
        loom.semaphore_give %17 : memref<64x512xf16>
        %21 = arith.muli %13, %c262144 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f10__d_d__block_size_064__block_size_14096__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 4096] on @L1 : memref<512x4096xf16>
        %15 = loom.semaphore_take %14 : memref<512x4096xf16> -> memref<512x4096xf16>
        %16 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %17 = loom.semaphore_take %16 : memref<64x512xf16> -> memref<64x512xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        %20 = arith.muli %13, %c32768 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x512xf16, strided<[512, 1], offset: ?>>, memref<64x512xf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<512x4096xf16, strided<[4096, 1]>>
        %cast = memref.cast %reinterpret_cast_0 : memref<512x4096xf16, strided<[4096, 1]>> to memref<512x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x4096xf16, strided<[4096, 1], offset: ?>>, memref<512x4096xf16>
        linalg.matmul ins(%17, %15 : memref<64x512xf16>, memref<512x4096xf16>) outs(%19 : memref<64x4096xf16>)
        loom.semaphore_give %15 : memref<512x4096xf16>
        loom.semaphore_give %17 : memref<64x512xf16>
        %21 = arith.muli %13, %c262144 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f10__d_a__block_size_064__block_size_14096__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 4096] on @L1 : memref<512x4096xf16>
        %15 = loom.semaphore_take %14 : memref<512x4096xf16> -> memref<512x4096xf16>
        %16 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %17 = loom.semaphore_take %16 : memref<64x512xf16> -> memref<64x512xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        %20 = arith.muli %13, %c32768 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x512xf16, strided<[512, 1], offset: ?>>, memref<64x512xf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<512x4096xf16, strided<[4096, 1]>>
        %cast = memref.cast %reinterpret_cast_0 : memref<512x4096xf16, strided<[4096, 1]>> to memref<512x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<512x4096xf16, strided<[4096, 1], offset: ?>>, memref<512x4096xf16>
        linalg.matmul ins(%17, %15 : memref<64x512xf16>, memref<512x4096xf16>) outs(%19 : memref<64x4096xf16>)
        loom.semaphore_give %15 : memref<512x4096xf16>
        loom.semaphore_give %17 : memref<64x512xf16>
        %21 = arith.muli %13, %c262144 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f01__d_d__block_size_0512__block_size_1512__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %13 = loom.semaphore_take %12 : memref<512x512xf16> -> memref<512x512xf16>
        %14 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %15 = loom.semaphore_take %14 : memref<512x512xf16> -> memref<512x512xf16>
        %16 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %17 = loom.semaphore_take %16 : memref<512x512xf16> -> memref<512x512xf16>
        linalg.fill ins(%cst : f16) outs(%17 : memref<512x512xf16>)
        %18 = arith.muli %arg3, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%18], sizes: [512, 512], strides: [512, 1] : memref<4096x512xf16> to memref<512x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x512xf16, strided<[512, 1], offset: ?>>, memref<512x512xf16>
        %19 = arith.muli %arg4, %c512 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%19], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %13 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x512xf16, strided<[4096, 1], offset: ?>>, memref<512x512xf16>
        linalg.matmul ins(%15, %13 : memref<512x512xf16>, memref<512x512xf16>) outs(%17 : memref<512x512xf16>)
        loom.semaphore_give %13 : memref<512x512xf16>
        loom.semaphore_give %15 : memref<512x512xf16>
        %20 = arith.muli %arg3, %c2097152 : index
        %21 = arith.addi %20, %19 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %17, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<512x512xf16>, memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %17 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f01__d_h__block_size_0512__block_size_1512__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %13 = loom.semaphore_take %12 : memref<512x512xf16> -> memref<512x512xf16>
        %14 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %15 = loom.semaphore_take %14 : memref<512x512xf16> -> memref<512x512xf16>
        %16 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %17 = loom.semaphore_take %16 : memref<512x512xf16> -> memref<512x512xf16>
        linalg.fill ins(%cst : f16) outs(%17 : memref<512x512xf16>)
        %18 = arith.muli %arg3, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%18], sizes: [512, 512], strides: [512, 1] : memref<4096x512xf16> to memref<512x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x512xf16, strided<[512, 1], offset: ?>>, memref<512x512xf16>
        %19 = arith.muli %arg4, %c512 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%19], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %13 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<512x512xf16, strided<[4096, 1], offset: ?>>, memref<512x512xf16>
        linalg.matmul ins(%15, %13 : memref<512x512xf16>, memref<512x512xf16>) outs(%17 : memref<512x512xf16>)
        loom.semaphore_give %13 : memref<512x512xf16>
        loom.semaphore_give %15 : memref<512x512xf16>
        %20 = arith.muli %arg3, %c2097152 : index
        %21 = arith.addi %20, %19 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %17, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<512x512xf16>, memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %17 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f01__v_d__block_size_0512__block_size_1512__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %13 = loom.semaphore_take %12 : memref<512x512xf16> -> memref<512x512xf16>
        %14 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %15 = loom.semaphore_take %14 : memref<512x512xf16> -> memref<512x512xf16>
        %16 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %17 = loom.semaphore_take %16 : memref<512x512xf16> -> memref<512x512xf16>
        linalg.fill ins(%cst : f16) outs(%17 : memref<512x512xf16>)
        %18 = arith.muli %arg3, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%18], sizes: [512, 512], strides: [512, 1] : memref<4096x512xf16> to memref<512x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<512x512xf16, strided<[512, 1], offset: ?>>, memref<512x512xf16>
        %19 = arith.muli %arg4, %c512 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%19], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %13 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x512xf16, strided<[4096, 1], offset: ?>>, memref<512x512xf16>
        linalg.matmul ins(%15, %13 : memref<512x512xf16>, memref<512x512xf16>) outs(%17 : memref<512x512xf16>)
        loom.semaphore_give %13 : memref<512x512xf16>
        loom.semaphore_give %15 : memref<512x512xf16>
        %20 = arith.muli %arg3, %c2097152 : index
        %21 = arith.addi %20, %19 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %17, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<512x512xf16>, memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %17 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f01__v_h__block_size_0512__block_size_1512__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %13 = loom.semaphore_take %12 : memref<512x512xf16> -> memref<512x512xf16>
        %14 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %15 = loom.semaphore_take %14 : memref<512x512xf16> -> memref<512x512xf16>
        %16 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %17 = loom.semaphore_take %16 : memref<512x512xf16> -> memref<512x512xf16>
        linalg.fill ins(%cst : f16) outs(%17 : memref<512x512xf16>)
        %18 = arith.muli %arg3, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%18], sizes: [512, 512], strides: [512, 1] : memref<4096x512xf16> to memref<512x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<512x512xf16, strided<[512, 1], offset: ?>>, memref<512x512xf16>
        %19 = arith.muli %arg4, %c512 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%19], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %13 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<512x512xf16, strided<[4096, 1], offset: ?>>, memref<512x512xf16>
        linalg.matmul ins(%15, %13 : memref<512x512xf16>, memref<512x512xf16>) outs(%17 : memref<512x512xf16>)
        loom.semaphore_give %13 : memref<512x512xf16>
        loom.semaphore_give %15 : memref<512x512xf16>
        %20 = arith.muli %arg3, %c2097152 : index
        %21 = arith.addi %20, %19 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %17, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<512x512xf16>, memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %17 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f10__d_d__block_size_0512__block_size_1512__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %13 = loom.semaphore_take %12 : memref<512x512xf16> -> memref<512x512xf16>
        %14 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %15 = loom.semaphore_take %14 : memref<512x512xf16> -> memref<512x512xf16>
        %16 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %17 = loom.semaphore_take %16 : memref<512x512xf16> -> memref<512x512xf16>
        linalg.fill ins(%cst : f16) outs(%17 : memref<512x512xf16>)
        %18 = arith.muli %arg3, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%18], sizes: [512, 512], strides: [512, 1] : memref<4096x512xf16> to memref<512x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x512xf16, strided<[512, 1], offset: ?>>, memref<512x512xf16>
        %19 = arith.muli %arg4, %c512 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%19], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %13 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x512xf16, strided<[4096, 1], offset: ?>>, memref<512x512xf16>
        linalg.matmul ins(%15, %13 : memref<512x512xf16>, memref<512x512xf16>) outs(%17 : memref<512x512xf16>)
        loom.semaphore_give %13 : memref<512x512xf16>
        loom.semaphore_give %15 : memref<512x512xf16>
        %20 = arith.muli %arg3, %c2097152 : index
        %21 = arith.addi %20, %19 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %17, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<512x512xf16>, memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %17 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f10__d_h__block_size_0512__block_size_1512__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %13 = loom.semaphore_take %12 : memref<512x512xf16> -> memref<512x512xf16>
        %14 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %15 = loom.semaphore_take %14 : memref<512x512xf16> -> memref<512x512xf16>
        %16 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %17 = loom.semaphore_take %16 : memref<512x512xf16> -> memref<512x512xf16>
        linalg.fill ins(%cst : f16) outs(%17 : memref<512x512xf16>)
        %18 = arith.muli %arg3, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%18], sizes: [512, 512], strides: [512, 1] : memref<4096x512xf16> to memref<512x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x512xf16, strided<[512, 1], offset: ?>>, memref<512x512xf16>
        %19 = arith.muli %arg4, %c512 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%19], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %13 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<512x512xf16, strided<[4096, 1], offset: ?>>, memref<512x512xf16>
        linalg.matmul ins(%15, %13 : memref<512x512xf16>, memref<512x512xf16>) outs(%17 : memref<512x512xf16>)
        loom.semaphore_give %13 : memref<512x512xf16>
        loom.semaphore_give %15 : memref<512x512xf16>
        %20 = arith.muli %arg3, %c2097152 : index
        %21 = arith.addi %20, %19 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %17, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<512x512xf16>, memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %17 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f10__v_d__block_size_0512__block_size_1512__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %13 = loom.semaphore_take %12 : memref<512x512xf16> -> memref<512x512xf16>
        %14 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %15 = loom.semaphore_take %14 : memref<512x512xf16> -> memref<512x512xf16>
        %16 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %17 = loom.semaphore_take %16 : memref<512x512xf16> -> memref<512x512xf16>
        linalg.fill ins(%cst : f16) outs(%17 : memref<512x512xf16>)
        %18 = arith.muli %arg3, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%18], sizes: [512, 512], strides: [512, 1] : memref<4096x512xf16> to memref<512x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<512x512xf16, strided<[512, 1], offset: ?>>, memref<512x512xf16>
        %19 = arith.muli %arg4, %c512 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%19], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %13 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x512xf16, strided<[4096, 1], offset: ?>>, memref<512x512xf16>
        linalg.matmul ins(%15, %13 : memref<512x512xf16>, memref<512x512xf16>) outs(%17 : memref<512x512xf16>)
        loom.semaphore_give %13 : memref<512x512xf16>
        loom.semaphore_give %15 : memref<512x512xf16>
        %20 = arith.muli %arg3, %c2097152 : index
        %21 = arith.addi %20, %19 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %17, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<512x512xf16>, memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %17 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f10__v_h__block_size_0512__block_size_1512__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %13 = loom.semaphore_take %12 : memref<512x512xf16> -> memref<512x512xf16>
        %14 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %15 = loom.semaphore_take %14 : memref<512x512xf16> -> memref<512x512xf16>
        %16 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %17 = loom.semaphore_take %16 : memref<512x512xf16> -> memref<512x512xf16>
        linalg.fill ins(%cst : f16) outs(%17 : memref<512x512xf16>)
        %18 = arith.muli %arg3, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%18], sizes: [512, 512], strides: [512, 1] : memref<4096x512xf16> to memref<512x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<512x512xf16, strided<[512, 1], offset: ?>>, memref<512x512xf16>
        %19 = arith.muli %arg4, %c512 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%19], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %13 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<512x512xf16, strided<[4096, 1], offset: ?>>, memref<512x512xf16>
        linalg.matmul ins(%15, %13 : memref<512x512xf16>, memref<512x512xf16>) outs(%17 : memref<512x512xf16>)
        loom.semaphore_give %13 : memref<512x512xf16>
        loom.semaphore_give %15 : memref<512x512xf16>
        %20 = arith.muli %arg3, %c2097152 : index
        %21 = arith.addi %20, %19 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %17, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<512x512xf16>, memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %17 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f01__d_d__block_size_0512__block_size_1512__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %13 = loom.semaphore_take %12 : memref<512x512xf16> -> memref<512x512xf16>
        %14 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %15 = loom.semaphore_take %14 : memref<512x512xf16> -> memref<512x512xf16>
        %16 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %17 = loom.semaphore_take %16 : memref<512x512xf16> -> memref<512x512xf16>
        linalg.fill ins(%cst : f16) outs(%17 : memref<512x512xf16>)
        %18 = arith.muli %arg3, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%18], sizes: [512, 512], strides: [512, 1] : memref<4096x512xf16> to memref<512x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x512xf16, strided<[512, 1], offset: ?>>, memref<512x512xf16>
        %19 = arith.muli %arg4, %c512 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%19], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %13 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x512xf16, strided<[4096, 1], offset: ?>>, memref<512x512xf16>
        linalg.matmul ins(%15, %13 : memref<512x512xf16>, memref<512x512xf16>) outs(%17 : memref<512x512xf16>)
        loom.semaphore_give %13 : memref<512x512xf16>
        loom.semaphore_give %15 : memref<512x512xf16>
        %20 = arith.muli %arg3, %c2097152 : index
        %21 = arith.addi %20, %19 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %17, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<512x512xf16>, memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %17 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f01__d_v__block_size_0512__block_size_1512__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %13 = loom.semaphore_take %12 : memref<512x512xf16> -> memref<512x512xf16>
        %14 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %15 = loom.semaphore_take %14 : memref<512x512xf16> -> memref<512x512xf16>
        %16 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %17 = loom.semaphore_take %16 : memref<512x512xf16> -> memref<512x512xf16>
        linalg.fill ins(%cst : f16) outs(%17 : memref<512x512xf16>)
        %18 = arith.muli %arg3, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%18], sizes: [512, 512], strides: [512, 1] : memref<4096x512xf16> to memref<512x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x512xf16, strided<[512, 1], offset: ?>>, memref<512x512xf16>
        %19 = arith.muli %arg4, %c512 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%19], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %13 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<512x512xf16, strided<[4096, 1], offset: ?>>, memref<512x512xf16>
        linalg.matmul ins(%15, %13 : memref<512x512xf16>, memref<512x512xf16>) outs(%17 : memref<512x512xf16>)
        loom.semaphore_give %13 : memref<512x512xf16>
        loom.semaphore_give %15 : memref<512x512xf16>
        %20 = arith.muli %arg3, %c2097152 : index
        %21 = arith.addi %20, %19 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %17, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<512x512xf16>, memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %17 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f01__h_d__block_size_0512__block_size_1512__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %13 = loom.semaphore_take %12 : memref<512x512xf16> -> memref<512x512xf16>
        %14 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %15 = loom.semaphore_take %14 : memref<512x512xf16> -> memref<512x512xf16>
        %16 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %17 = loom.semaphore_take %16 : memref<512x512xf16> -> memref<512x512xf16>
        linalg.fill ins(%cst : f16) outs(%17 : memref<512x512xf16>)
        %18 = arith.muli %arg3, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%18], sizes: [512, 512], strides: [512, 1] : memref<4096x512xf16> to memref<512x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<512x512xf16, strided<[512, 1], offset: ?>>, memref<512x512xf16>
        %19 = arith.muli %arg4, %c512 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%19], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %13 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x512xf16, strided<[4096, 1], offset: ?>>, memref<512x512xf16>
        linalg.matmul ins(%15, %13 : memref<512x512xf16>, memref<512x512xf16>) outs(%17 : memref<512x512xf16>)
        loom.semaphore_give %13 : memref<512x512xf16>
        loom.semaphore_give %15 : memref<512x512xf16>
        %20 = arith.muli %arg3, %c2097152 : index
        %21 = arith.addi %20, %19 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %17, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<512x512xf16>, memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %17 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f10__d_d__block_size_0512__block_size_1512__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %13 = loom.semaphore_take %12 : memref<512x512xf16> -> memref<512x512xf16>
        %14 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %15 = loom.semaphore_take %14 : memref<512x512xf16> -> memref<512x512xf16>
        %16 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %17 = loom.semaphore_take %16 : memref<512x512xf16> -> memref<512x512xf16>
        linalg.fill ins(%cst : f16) outs(%17 : memref<512x512xf16>)
        %18 = arith.muli %arg3, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%18], sizes: [512, 512], strides: [512, 1] : memref<4096x512xf16> to memref<512x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x512xf16, strided<[512, 1], offset: ?>>, memref<512x512xf16>
        %19 = arith.muli %arg4, %c512 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%19], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %13 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x512xf16, strided<[4096, 1], offset: ?>>, memref<512x512xf16>
        linalg.matmul ins(%15, %13 : memref<512x512xf16>, memref<512x512xf16>) outs(%17 : memref<512x512xf16>)
        loom.semaphore_give %13 : memref<512x512xf16>
        loom.semaphore_give %15 : memref<512x512xf16>
        %20 = arith.muli %arg3, %c2097152 : index
        %21 = arith.addi %20, %19 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %17, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<512x512xf16>, memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %17 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f10__d_v__block_size_0512__block_size_1512__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %13 = loom.semaphore_take %12 : memref<512x512xf16> -> memref<512x512xf16>
        %14 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %15 = loom.semaphore_take %14 : memref<512x512xf16> -> memref<512x512xf16>
        %16 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %17 = loom.semaphore_take %16 : memref<512x512xf16> -> memref<512x512xf16>
        linalg.fill ins(%cst : f16) outs(%17 : memref<512x512xf16>)
        %18 = arith.muli %arg3, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%18], sizes: [512, 512], strides: [512, 1] : memref<4096x512xf16> to memref<512x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x512xf16, strided<[512, 1], offset: ?>>, memref<512x512xf16>
        %19 = arith.muli %arg4, %c512 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%19], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %13 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<512x512xf16, strided<[4096, 1], offset: ?>>, memref<512x512xf16>
        linalg.matmul ins(%15, %13 : memref<512x512xf16>, memref<512x512xf16>) outs(%17 : memref<512x512xf16>)
        loom.semaphore_give %13 : memref<512x512xf16>
        loom.semaphore_give %15 : memref<512x512xf16>
        %20 = arith.muli %arg3, %c2097152 : index
        %21 = arith.addi %20, %19 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %17, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<512x512xf16>, memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %17 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f10__h_d__block_size_0512__block_size_1512__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %13 = loom.semaphore_take %12 : memref<512x512xf16> -> memref<512x512xf16>
        %14 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %15 = loom.semaphore_take %14 : memref<512x512xf16> -> memref<512x512xf16>
        %16 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %17 = loom.semaphore_take %16 : memref<512x512xf16> -> memref<512x512xf16>
        linalg.fill ins(%cst : f16) outs(%17 : memref<512x512xf16>)
        %18 = arith.muli %arg3, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%18], sizes: [512, 512], strides: [512, 1] : memref<4096x512xf16> to memref<512x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<512x512xf16, strided<[512, 1], offset: ?>>, memref<512x512xf16>
        %19 = arith.muli %arg4, %c512 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%19], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %13 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x512xf16, strided<[4096, 1], offset: ?>>, memref<512x512xf16>
        linalg.matmul ins(%15, %13 : memref<512x512xf16>, memref<512x512xf16>) outs(%17 : memref<512x512xf16>)
        loom.semaphore_give %13 : memref<512x512xf16>
        loom.semaphore_give %15 : memref<512x512xf16>
        %20 = arith.muli %arg3, %c2097152 : index
        %21 = arith.addi %20, %19 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %17, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<512x512xf16>, memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %17 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f01__d_d__block_size_04096__block_size_164__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %15 = loom.semaphore_take %14 : memref<512x64xf16> -> memref<512x64xf16>
        %16 = loom.alloc [4096, 512] on @L1 : memref<4096x512xf16>
        %17 = loom.semaphore_take %16 : memref<4096x512xf16> -> memref<4096x512xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096, 512], strides: [512, 1] : memref<4096x512xf16> to memref<4096x512xf16, strided<[512, 1]>>
        %cast = memref.cast %reinterpret_cast : memref<4096x512xf16, strided<[512, 1]>> to memref<4096x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<4096x512xf16, strided<[512, 1], offset: ?>>, memref<4096x512xf16>
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x64xf16, strided<[4096, 1], offset: ?>>, memref<512x64xf16>
        linalg.matmul ins(%17, %15 : memref<4096x512xf16>, memref<512x64xf16>) outs(%19 : memref<4096x64xf16>)
        loom.semaphore_give %15 : memref<512x64xf16>
        loom.semaphore_give %17 : memref<4096x512xf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f01__a_d__block_size_04096__block_size_164__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %15 = loom.semaphore_take %14 : memref<512x64xf16> -> memref<512x64xf16>
        %16 = loom.alloc [4096, 512] on @L1 : memref<4096x512xf16>
        %17 = loom.semaphore_take %16 : memref<4096x512xf16> -> memref<4096x512xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096, 512], strides: [512, 1] : memref<4096x512xf16> to memref<4096x512xf16, strided<[512, 1]>>
        %cast = memref.cast %reinterpret_cast : memref<4096x512xf16, strided<[512, 1]>> to memref<4096x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<4096x512xf16, strided<[512, 1], offset: ?>>, memref<4096x512xf16>
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x64xf16, strided<[4096, 1], offset: ?>>, memref<512x64xf16>
        linalg.matmul ins(%17, %15 : memref<4096x512xf16>, memref<512x64xf16>) outs(%19 : memref<4096x64xf16>)
        loom.semaphore_give %15 : memref<512x64xf16>
        loom.semaphore_give %17 : memref<4096x512xf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f01__h_d__block_size_04096__block_size_164__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %15 = loom.semaphore_take %14 : memref<512x64xf16> -> memref<512x64xf16>
        %16 = loom.alloc [4096, 512] on @L1 : memref<4096x512xf16>
        %17 = loom.semaphore_take %16 : memref<4096x512xf16> -> memref<4096x512xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096, 512], strides: [512, 1] : memref<4096x512xf16> to memref<4096x512xf16, strided<[512, 1]>>
        %cast = memref.cast %reinterpret_cast : memref<4096x512xf16, strided<[512, 1]>> to memref<4096x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<4096x512xf16, strided<[512, 1], offset: ?>>, memref<4096x512xf16>
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x64xf16, strided<[4096, 1], offset: ?>>, memref<512x64xf16>
        linalg.matmul ins(%17, %15 : memref<4096x512xf16>, memref<512x64xf16>) outs(%19 : memref<4096x64xf16>)
        loom.semaphore_give %15 : memref<512x64xf16>
        loom.semaphore_give %17 : memref<4096x512xf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f01__v_d__block_size_04096__block_size_164__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %15 = loom.semaphore_take %14 : memref<512x64xf16> -> memref<512x64xf16>
        %16 = loom.alloc [4096, 512] on @L1 : memref<4096x512xf16>
        %17 = loom.semaphore_take %16 : memref<4096x512xf16> -> memref<4096x512xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096, 512], strides: [512, 1] : memref<4096x512xf16> to memref<4096x512xf16, strided<[512, 1]>>
        %cast = memref.cast %reinterpret_cast : memref<4096x512xf16, strided<[512, 1]>> to memref<4096x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<4096x512xf16, strided<[512, 1], offset: ?>>, memref<4096x512xf16>
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x64xf16, strided<[4096, 1], offset: ?>>, memref<512x64xf16>
        linalg.matmul ins(%17, %15 : memref<4096x512xf16>, memref<512x64xf16>) outs(%19 : memref<4096x64xf16>)
        loom.semaphore_give %15 : memref<512x64xf16>
        loom.semaphore_give %17 : memref<4096x512xf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f10__d_d__block_size_04096__block_size_164__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %15 = loom.semaphore_take %14 : memref<512x64xf16> -> memref<512x64xf16>
        %16 = loom.alloc [4096, 512] on @L1 : memref<4096x512xf16>
        %17 = loom.semaphore_take %16 : memref<4096x512xf16> -> memref<4096x512xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096, 512], strides: [512, 1] : memref<4096x512xf16> to memref<4096x512xf16, strided<[512, 1]>>
        %cast = memref.cast %reinterpret_cast : memref<4096x512xf16, strided<[512, 1]>> to memref<4096x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<4096x512xf16, strided<[512, 1], offset: ?>>, memref<4096x512xf16>
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x64xf16, strided<[4096, 1], offset: ?>>, memref<512x64xf16>
        linalg.matmul ins(%17, %15 : memref<4096x512xf16>, memref<512x64xf16>) outs(%19 : memref<4096x64xf16>)
        loom.semaphore_give %15 : memref<512x64xf16>
        loom.semaphore_give %17 : memref<4096x512xf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f10__a_d__block_size_04096__block_size_164__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %15 = loom.semaphore_take %14 : memref<512x64xf16> -> memref<512x64xf16>
        %16 = loom.alloc [4096, 512] on @L1 : memref<4096x512xf16>
        %17 = loom.semaphore_take %16 : memref<4096x512xf16> -> memref<4096x512xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096, 512], strides: [512, 1] : memref<4096x512xf16> to memref<4096x512xf16, strided<[512, 1]>>
        %cast = memref.cast %reinterpret_cast : memref<4096x512xf16, strided<[512, 1]>> to memref<4096x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<4096x512xf16, strided<[512, 1], offset: ?>>, memref<4096x512xf16>
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x64xf16, strided<[4096, 1], offset: ?>>, memref<512x64xf16>
        linalg.matmul ins(%17, %15 : memref<4096x512xf16>, memref<512x64xf16>) outs(%19 : memref<4096x64xf16>)
        loom.semaphore_give %15 : memref<512x64xf16>
        loom.semaphore_give %17 : memref<4096x512xf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f10__v_d__block_size_04096__block_size_164__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %15 = loom.semaphore_take %14 : memref<512x64xf16> -> memref<512x64xf16>
        %16 = loom.alloc [4096, 512] on @L1 : memref<4096x512xf16>
        %17 = loom.semaphore_take %16 : memref<4096x512xf16> -> memref<4096x512xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096, 512], strides: [512, 1] : memref<4096x512xf16> to memref<4096x512xf16, strided<[512, 1]>>
        %cast = memref.cast %reinterpret_cast : memref<4096x512xf16, strided<[512, 1]>> to memref<4096x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<4096x512xf16, strided<[512, 1], offset: ?>>, memref<4096x512xf16>
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x64xf16, strided<[4096, 1], offset: ?>>, memref<512x64xf16>
        linalg.matmul ins(%17, %15 : memref<4096x512xf16>, memref<512x64xf16>) outs(%19 : memref<4096x64xf16>)
        loom.semaphore_give %15 : memref<512x64xf16>
        loom.semaphore_give %17 : memref<4096x512xf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f01__d_d__block_size_04096__block_size_164__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %15 = loom.semaphore_take %14 : memref<512x64xf16> -> memref<512x64xf16>
        %16 = loom.alloc [4096, 512] on @L1 : memref<4096x512xf16>
        %17 = loom.semaphore_take %16 : memref<4096x512xf16> -> memref<4096x512xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096, 512], strides: [512, 1] : memref<4096x512xf16> to memref<4096x512xf16, strided<[512, 1]>>
        %cast = memref.cast %reinterpret_cast : memref<4096x512xf16, strided<[512, 1]>> to memref<4096x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<4096x512xf16, strided<[512, 1], offset: ?>>, memref<4096x512xf16>
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x64xf16, strided<[4096, 1], offset: ?>>, memref<512x64xf16>
        linalg.matmul ins(%17, %15 : memref<4096x512xf16>, memref<512x64xf16>) outs(%19 : memref<4096x64xf16>)
        loom.semaphore_give %15 : memref<512x64xf16>
        loom.semaphore_give %17 : memref<4096x512xf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f01__h_d__block_size_04096__block_size_164__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %15 = loom.semaphore_take %14 : memref<512x64xf16> -> memref<512x64xf16>
        %16 = loom.alloc [4096, 512] on @L1 : memref<4096x512xf16>
        %17 = loom.semaphore_take %16 : memref<4096x512xf16> -> memref<4096x512xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096, 512], strides: [512, 1] : memref<4096x512xf16> to memref<4096x512xf16, strided<[512, 1]>>
        %cast = memref.cast %reinterpret_cast : memref<4096x512xf16, strided<[512, 1]>> to memref<4096x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<4096x512xf16, strided<[512, 1], offset: ?>>, memref<4096x512xf16>
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x64xf16, strided<[4096, 1], offset: ?>>, memref<512x64xf16>
        linalg.matmul ins(%17, %15 : memref<4096x512xf16>, memref<512x64xf16>) outs(%19 : memref<4096x64xf16>)
        loom.semaphore_give %15 : memref<512x64xf16>
        loom.semaphore_give %17 : memref<4096x512xf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f01__v_d__block_size_04096__block_size_164__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %15 = loom.semaphore_take %14 : memref<512x64xf16> -> memref<512x64xf16>
        %16 = loom.alloc [4096, 512] on @L1 : memref<4096x512xf16>
        %17 = loom.semaphore_take %16 : memref<4096x512xf16> -> memref<4096x512xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096, 512], strides: [512, 1] : memref<4096x512xf16> to memref<4096x512xf16, strided<[512, 1]>>
        %cast = memref.cast %reinterpret_cast : memref<4096x512xf16, strided<[512, 1]>> to memref<4096x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<4096x512xf16, strided<[512, 1], offset: ?>>, memref<4096x512xf16>
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x64xf16, strided<[4096, 1], offset: ?>>, memref<512x64xf16>
        linalg.matmul ins(%17, %15 : memref<4096x512xf16>, memref<512x64xf16>) outs(%19 : memref<4096x64xf16>)
        loom.semaphore_give %15 : memref<512x64xf16>
        loom.semaphore_give %17 : memref<4096x512xf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f10__d_d__block_size_04096__block_size_164__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %15 = loom.semaphore_take %14 : memref<512x64xf16> -> memref<512x64xf16>
        %16 = loom.alloc [4096, 512] on @L1 : memref<4096x512xf16>
        %17 = loom.semaphore_take %16 : memref<4096x512xf16> -> memref<4096x512xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096, 512], strides: [512, 1] : memref<4096x512xf16> to memref<4096x512xf16, strided<[512, 1]>>
        %cast = memref.cast %reinterpret_cast : memref<4096x512xf16, strided<[512, 1]>> to memref<4096x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<4096x512xf16, strided<[512, 1], offset: ?>>, memref<4096x512xf16>
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x64xf16, strided<[4096, 1], offset: ?>>, memref<512x64xf16>
        linalg.matmul ins(%17, %15 : memref<4096x512xf16>, memref<512x64xf16>) outs(%19 : memref<4096x64xf16>)
        loom.semaphore_give %15 : memref<512x64xf16>
        loom.semaphore_give %17 : memref<4096x512xf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f10__a_d__block_size_04096__block_size_164__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %15 = loom.semaphore_take %14 : memref<512x64xf16> -> memref<512x64xf16>
        %16 = loom.alloc [4096, 512] on @L1 : memref<4096x512xf16>
        %17 = loom.semaphore_take %16 : memref<4096x512xf16> -> memref<4096x512xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096, 512], strides: [512, 1] : memref<4096x512xf16> to memref<4096x512xf16, strided<[512, 1]>>
        %cast = memref.cast %reinterpret_cast : memref<4096x512xf16, strided<[512, 1]>> to memref<4096x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<4096x512xf16, strided<[512, 1], offset: ?>>, memref<4096x512xf16>
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x64xf16, strided<[4096, 1], offset: ?>>, memref<512x64xf16>
        linalg.matmul ins(%17, %15 : memref<4096x512xf16>, memref<512x64xf16>) outs(%19 : memref<4096x64xf16>)
        loom.semaphore_give %15 : memref<512x64xf16>
        loom.semaphore_give %17 : memref<4096x512xf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f10__h_d__block_size_04096__block_size_164__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %15 = loom.semaphore_take %14 : memref<512x64xf16> -> memref<512x64xf16>
        %16 = loom.alloc [4096, 512] on @L1 : memref<4096x512xf16>
        %17 = loom.semaphore_take %16 : memref<4096x512xf16> -> memref<4096x512xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096, 512], strides: [512, 1] : memref<4096x512xf16> to memref<4096x512xf16, strided<[512, 1]>>
        %cast = memref.cast %reinterpret_cast : memref<4096x512xf16, strided<[512, 1]>> to memref<4096x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<4096x512xf16, strided<[512, 1], offset: ?>>, memref<4096x512xf16>
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x64xf16, strided<[4096, 1], offset: ?>>, memref<512x64xf16>
        linalg.matmul ins(%17, %15 : memref<4096x512xf16>, memref<512x64xf16>) outs(%19 : memref<4096x64xf16>)
        loom.semaphore_give %15 : memref<512x64xf16>
        loom.semaphore_give %17 : memref<4096x512xf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f10__v_d__block_size_04096__block_size_164__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %15 = loom.semaphore_take %14 : memref<512x64xf16> -> memref<512x64xf16>
        %16 = loom.alloc [4096, 512] on @L1 : memref<4096x512xf16>
        %17 = loom.semaphore_take %16 : memref<4096x512xf16> -> memref<4096x512xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096, 512], strides: [512, 1] : memref<4096x512xf16> to memref<4096x512xf16, strided<[512, 1]>>
        %cast = memref.cast %reinterpret_cast : memref<4096x512xf16, strided<[512, 1]>> to memref<4096x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<4096x512xf16, strided<[512, 1], offset: ?>>, memref<4096x512xf16>
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x64xf16, strided<[4096, 1], offset: ?>>, memref<512x64xf16>
        linalg.matmul ins(%17, %15 : memref<4096x512xf16>, memref<512x64xf16>) outs(%19 : memref<4096x64xf16>)
        loom.semaphore_give %15 : memref<512x64xf16>
        loom.semaphore_give %17 : memref<4096x512xf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
}
