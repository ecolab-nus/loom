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
    func.func @matmul__d0i0_d1i0__f01__d_d__BK512__BM64__BN4096(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %12, %c64 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.divsi %c4095, %13 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg5, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %23, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %arg6, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %23, %12 : index
            %33 = arith.muli %arg6, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_a__BK512__BM64__BN4096(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %12, %c64 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.divsi %c4095, %13 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg5, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %23, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %arg6, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %23, %12 : index
            %33 = arith.muli %arg6, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_h__BK64__BM64__BN4096(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %12, %c64 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.divsi %c4095, %13 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg5, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %23, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %arg6, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %23, %12 : index
            %33 = arith.muli %arg6, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_v__BK256__BM64__BN1024(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %12, %c64 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.divsi %c4095, %13 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg5, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %23, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %arg6, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %23, %12 : index
            %33 = arith.muli %arg6, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_d__BK512__BM64__BN4096(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.divsi %c4095, %13 : index
        %16 = arith.addi %15, %c1 : index
        scf.for %arg5 = %c0 to %16 step %c1 {
          %17 = arith.muli %12, %c64 overflow<nsw> : index
          %18 = arith.divsi %c4095, %17 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg6, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %23, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %arg5, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %23, %12 : index
            %33 = arith.muli %arg5, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_a__BK512__BM64__BN4096(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.divsi %c4095, %13 : index
        %16 = arith.addi %15, %c1 : index
        scf.for %arg5 = %c0 to %16 step %c1 {
          %17 = arith.muli %12, %c64 overflow<nsw> : index
          %18 = arith.divsi %c4095, %17 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg6, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %23, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %arg5, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %23, %12 : index
            %33 = arith.muli %arg5, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_h__BK32__BM64__BN4096(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.divsi %c4095, %13 : index
        %16 = arith.addi %15, %c1 : index
        scf.for %arg5 = %c0 to %16 step %c1 {
          %17 = arith.muli %12, %c64 overflow<nsw> : index
          %18 = arith.divsi %c4095, %17 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg6, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %23, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %arg5, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %23, %12 : index
            %33 = arith.muli %arg5, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_v__BK128__BM64__BN4096(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.divsi %c4095, %13 : index
        %16 = arith.addi %15, %c1 : index
        scf.for %arg5 = %c0 to %16 step %c1 {
          %17 = arith.muli %12, %c64 overflow<nsw> : index
          %18 = arith.divsi %c4095, %17 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg6, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %23, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %arg5, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %23, %12 : index
            %33 = arith.muli %arg5, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f01__d_d__BK512__BM64__BN4096(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %12, %c64 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.divsi %c4095, %13 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg5, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %23, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %arg6, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %23, %12 : index
            %33 = arith.muli %arg6, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f01__d_a__BK512__BM64__BN4096(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %12, %c64 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.divsi %c4095, %13 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg5, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %23, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %arg6, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %23, %12 : index
            %33 = arith.muli %arg6, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f01__d_h__BK64__BM64__BN4096(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %12, %c64 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.divsi %c4095, %13 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg5, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %23, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %arg6, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %23, %12 : index
            %33 = arith.muli %arg6, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f01__d_v__BK256__BM64__BN1024(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %12, %c64 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.divsi %c4095, %13 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg5, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %23, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %arg6, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %23, %12 : index
            %33 = arith.muli %arg6, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f10__d_d__BK512__BM64__BN4096(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.divsi %c4095, %13 : index
        %16 = arith.addi %15, %c1 : index
        scf.for %arg5 = %c0 to %16 step %c1 {
          %17 = arith.muli %12, %c64 overflow<nsw> : index
          %18 = arith.divsi %c4095, %17 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg6, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %23, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %arg5, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %23, %12 : index
            %33 = arith.muli %arg5, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f10__d_a__BK512__BM64__BN4096(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.divsi %c4095, %13 : index
        %16 = arith.addi %15, %c1 : index
        scf.for %arg5 = %c0 to %16 step %c1 {
          %17 = arith.muli %12, %c64 overflow<nsw> : index
          %18 = arith.divsi %c4095, %17 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg6, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %23, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %arg5, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %23, %12 : index
            %33 = arith.muli %arg5, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f10__d_h__BK32__BM64__BN4096(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.divsi %c4095, %13 : index
        %16 = arith.addi %15, %c1 : index
        scf.for %arg5 = %c0 to %16 step %c1 {
          %17 = arith.muli %12, %c64 overflow<nsw> : index
          %18 = arith.divsi %c4095, %17 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg6, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %23, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %arg5, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %23, %12 : index
            %33 = arith.muli %arg5, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f10__d_v__BK128__BM64__BN4096(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.divsi %c4095, %13 : index
        %16 = arith.addi %15, %c1 : index
        scf.for %arg5 = %c0 to %16 step %c1 {
          %17 = arith.muli %12, %c64 overflow<nsw> : index
          %18 = arith.divsi %c4095, %17 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg6, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %23, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %arg5, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %23, %12 : index
            %33 = arith.muli %arg5, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__d_d__BK32__BM512__BN256(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %12, %c8 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.muli %13, %c8 overflow<nsw> : index
          %19 = arith.divsi %c4095, %18 : index
          %20 = arith.addi %19, %c1 : index
          scf.for %arg6 = %c0 to %20 step %c1 {
            %21 = arith.muli %arg5, %c8 overflow<nsw> : index
            %22 = arith.addi %arg3, %21 : index
            %23 = arith.muli %arg6, %c8 overflow<nsw> : index
            %24 = arith.addi %arg4, %23 : index
            %25 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %26 = loom.semaphore_take %25 : memref<?x?xf32> -> memref<?x?xf32>
            %27 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %28 = loom.semaphore_take %27 : memref<?x?xf32> -> memref<?x?xf32>
            %29 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%30 : memref<?x?xf32>)
            %31 = arith.divsi %c511, %14 : index
            %32 = arith.addi %31, %c1 : index
            scf.for %arg7 = %c0 to %32 step %c1 {
              %37 = arith.muli %22, %12 : index
              %38 = arith.muli %arg7, %14 : index
              %39 = arith.muli %37, %c512 overflow<nsw> : index
              %40 = arith.addi %39, %38 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%40], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %28 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %41 = arith.muli %24, %13 : index
              %42 = arith.muli %38, %c4096 overflow<nsw> : index
              %43 = arith.addi %42, %41 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %26 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%28, %26 : memref<?x?xf32>, memref<?x?xf32>) outs(%30 : memref<?x?xf32>)
              loom.semaphore_give %26 : memref<?x?xf32>
              loom.semaphore_give %28 : memref<?x?xf32>
            }
            %33 = arith.muli %22, %12 : index
            %34 = arith.muli %24, %13 : index
            %35 = arith.muli %33, %c4096 overflow<nsw> : index
            %36 = arith.addi %35, %34 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %30, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %30 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__d_h__BK64__BM512__BN128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %12, %c8 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.muli %13, %c8 overflow<nsw> : index
          %19 = arith.divsi %c4095, %18 : index
          %20 = arith.addi %19, %c1 : index
          scf.for %arg6 = %c0 to %20 step %c1 {
            %21 = arith.muli %arg5, %c8 overflow<nsw> : index
            %22 = arith.addi %arg3, %21 : index
            %23 = arith.muli %arg6, %c8 overflow<nsw> : index
            %24 = arith.addi %arg4, %23 : index
            %25 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %26 = loom.semaphore_take %25 : memref<?x?xf32> -> memref<?x?xf32>
            %27 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %28 = loom.semaphore_take %27 : memref<?x?xf32> -> memref<?x?xf32>
            %29 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%30 : memref<?x?xf32>)
            %31 = arith.divsi %c511, %14 : index
            %32 = arith.addi %31, %c1 : index
            scf.for %arg7 = %c0 to %32 step %c1 {
              %37 = arith.muli %22, %12 : index
              %38 = arith.muli %arg7, %14 : index
              %39 = arith.muli %37, %c512 overflow<nsw> : index
              %40 = arith.addi %39, %38 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%40], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %28 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %41 = arith.muli %24, %13 : index
              %42 = arith.muli %38, %c4096 overflow<nsw> : index
              %43 = arith.addi %42, %41 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %26 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%28, %26 : memref<?x?xf32>, memref<?x?xf32>) outs(%30 : memref<?x?xf32>)
              loom.semaphore_give %26 : memref<?x?xf32>
              loom.semaphore_give %28 : memref<?x?xf32>
            }
            %33 = arith.muli %22, %12 : index
            %34 = arith.muli %24, %13 : index
            %35 = arith.muli %33, %c4096 overflow<nsw> : index
            %36 = arith.addi %35, %34 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %30, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %30 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__v_d__BK512__BM256__BN512(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %12, %c8 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.muli %13, %c8 overflow<nsw> : index
          %19 = arith.divsi %c4095, %18 : index
          %20 = arith.addi %19, %c1 : index
          scf.for %arg6 = %c0 to %20 step %c1 {
            %21 = arith.muli %arg5, %c8 overflow<nsw> : index
            %22 = arith.addi %arg3, %21 : index
            %23 = arith.muli %arg6, %c8 overflow<nsw> : index
            %24 = arith.addi %arg4, %23 : index
            %25 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %26 = loom.semaphore_take %25 : memref<?x?xf32> -> memref<?x?xf32>
            %27 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %28 = loom.semaphore_take %27 : memref<?x?xf32> -> memref<?x?xf32>
            %29 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%30 : memref<?x?xf32>)
            %31 = arith.divsi %c511, %14 : index
            %32 = arith.addi %31, %c1 : index
            scf.for %arg7 = %c0 to %32 step %c1 {
              %37 = arith.muli %22, %12 : index
              %38 = arith.muli %arg7, %14 : index
              %39 = arith.muli %37, %c512 overflow<nsw> : index
              %40 = arith.addi %39, %38 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%40], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %28 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %41 = arith.muli %24, %13 : index
              %42 = arith.muli %38, %c4096 overflow<nsw> : index
              %43 = arith.addi %42, %41 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %26 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%28, %26 : memref<?x?xf32>, memref<?x?xf32>) outs(%30 : memref<?x?xf32>)
              loom.semaphore_give %26 : memref<?x?xf32>
              loom.semaphore_give %28 : memref<?x?xf32>
            }
            %33 = arith.muli %22, %12 : index
            %34 = arith.muli %24, %13 : index
            %35 = arith.muli %33, %c4096 overflow<nsw> : index
            %36 = arith.addi %35, %34 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %30, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %30 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__v_h__BK256__BM128__BN512(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %12, %c8 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.muli %13, %c8 overflow<nsw> : index
          %19 = arith.divsi %c4095, %18 : index
          %20 = arith.addi %19, %c1 : index
          scf.for %arg6 = %c0 to %20 step %c1 {
            %21 = arith.muli %arg5, %c8 overflow<nsw> : index
            %22 = arith.addi %arg3, %21 : index
            %23 = arith.muli %arg6, %c8 overflow<nsw> : index
            %24 = arith.addi %arg4, %23 : index
            %25 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %26 = loom.semaphore_take %25 : memref<?x?xf32> -> memref<?x?xf32>
            %27 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %28 = loom.semaphore_take %27 : memref<?x?xf32> -> memref<?x?xf32>
            %29 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%30 : memref<?x?xf32>)
            %31 = arith.divsi %c511, %14 : index
            %32 = arith.addi %31, %c1 : index
            scf.for %arg7 = %c0 to %32 step %c1 {
              %37 = arith.muli %22, %12 : index
              %38 = arith.muli %arg7, %14 : index
              %39 = arith.muli %37, %c512 overflow<nsw> : index
              %40 = arith.addi %39, %38 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%40], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %28 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %41 = arith.muli %24, %13 : index
              %42 = arith.muli %38, %c4096 overflow<nsw> : index
              %43 = arith.addi %42, %41 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %26 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%28, %26 : memref<?x?xf32>, memref<?x?xf32>) outs(%30 : memref<?x?xf32>)
              loom.semaphore_give %26 : memref<?x?xf32>
              loom.semaphore_give %28 : memref<?x?xf32>
            }
            %33 = arith.muli %22, %12 : index
            %34 = arith.muli %24, %13 : index
            %35 = arith.muli %33, %c4096 overflow<nsw> : index
            %36 = arith.addi %35, %34 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %30, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %30 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__d_d__BK32__BM512__BN256(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %13, %c8 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.muli %12, %c8 overflow<nsw> : index
          %19 = arith.divsi %c4095, %18 : index
          %20 = arith.addi %19, %c1 : index
          scf.for %arg6 = %c0 to %20 step %c1 {
            %21 = arith.muli %arg6, %c8 overflow<nsw> : index
            %22 = arith.addi %arg3, %21 : index
            %23 = arith.muli %arg5, %c8 overflow<nsw> : index
            %24 = arith.addi %arg4, %23 : index
            %25 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %26 = loom.semaphore_take %25 : memref<?x?xf32> -> memref<?x?xf32>
            %27 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %28 = loom.semaphore_take %27 : memref<?x?xf32> -> memref<?x?xf32>
            %29 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%30 : memref<?x?xf32>)
            %31 = arith.divsi %c511, %14 : index
            %32 = arith.addi %31, %c1 : index
            scf.for %arg7 = %c0 to %32 step %c1 {
              %37 = arith.muli %22, %12 : index
              %38 = arith.muli %arg7, %14 : index
              %39 = arith.muli %37, %c512 overflow<nsw> : index
              %40 = arith.addi %39, %38 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%40], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %28 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %41 = arith.muli %24, %13 : index
              %42 = arith.muli %38, %c4096 overflow<nsw> : index
              %43 = arith.addi %42, %41 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %26 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%28, %26 : memref<?x?xf32>, memref<?x?xf32>) outs(%30 : memref<?x?xf32>)
              loom.semaphore_give %26 : memref<?x?xf32>
              loom.semaphore_give %28 : memref<?x?xf32>
            }
            %33 = arith.muli %22, %12 : index
            %34 = arith.muli %24, %13 : index
            %35 = arith.muli %33, %c4096 overflow<nsw> : index
            %36 = arith.addi %35, %34 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %30, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %30 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__d_h__BK64__BM128__BN512(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %13, %c8 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.muli %12, %c8 overflow<nsw> : index
          %19 = arith.divsi %c4095, %18 : index
          %20 = arith.addi %19, %c1 : index
          scf.for %arg6 = %c0 to %20 step %c1 {
            %21 = arith.muli %arg6, %c8 overflow<nsw> : index
            %22 = arith.addi %arg3, %21 : index
            %23 = arith.muli %arg5, %c8 overflow<nsw> : index
            %24 = arith.addi %arg4, %23 : index
            %25 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %26 = loom.semaphore_take %25 : memref<?x?xf32> -> memref<?x?xf32>
            %27 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %28 = loom.semaphore_take %27 : memref<?x?xf32> -> memref<?x?xf32>
            %29 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%30 : memref<?x?xf32>)
            %31 = arith.divsi %c511, %14 : index
            %32 = arith.addi %31, %c1 : index
            scf.for %arg7 = %c0 to %32 step %c1 {
              %37 = arith.muli %22, %12 : index
              %38 = arith.muli %arg7, %14 : index
              %39 = arith.muli %37, %c512 overflow<nsw> : index
              %40 = arith.addi %39, %38 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%40], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %28 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %41 = arith.muli %24, %13 : index
              %42 = arith.muli %38, %c4096 overflow<nsw> : index
              %43 = arith.addi %42, %41 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %26 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%28, %26 : memref<?x?xf32>, memref<?x?xf32>) outs(%30 : memref<?x?xf32>)
              loom.semaphore_give %26 : memref<?x?xf32>
              loom.semaphore_give %28 : memref<?x?xf32>
            }
            %33 = arith.muli %22, %12 : index
            %34 = arith.muli %24, %13 : index
            %35 = arith.muli %33, %c4096 overflow<nsw> : index
            %36 = arith.addi %35, %34 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %30, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %30 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__v_d__BK512__BM256__BN512(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %13, %c8 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.muli %12, %c8 overflow<nsw> : index
          %19 = arith.divsi %c4095, %18 : index
          %20 = arith.addi %19, %c1 : index
          scf.for %arg6 = %c0 to %20 step %c1 {
            %21 = arith.muli %arg6, %c8 overflow<nsw> : index
            %22 = arith.addi %arg3, %21 : index
            %23 = arith.muli %arg5, %c8 overflow<nsw> : index
            %24 = arith.addi %arg4, %23 : index
            %25 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %26 = loom.semaphore_take %25 : memref<?x?xf32> -> memref<?x?xf32>
            %27 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %28 = loom.semaphore_take %27 : memref<?x?xf32> -> memref<?x?xf32>
            %29 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%30 : memref<?x?xf32>)
            %31 = arith.divsi %c511, %14 : index
            %32 = arith.addi %31, %c1 : index
            scf.for %arg7 = %c0 to %32 step %c1 {
              %37 = arith.muli %22, %12 : index
              %38 = arith.muli %arg7, %14 : index
              %39 = arith.muli %37, %c512 overflow<nsw> : index
              %40 = arith.addi %39, %38 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%40], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %28 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %41 = arith.muli %24, %13 : index
              %42 = arith.muli %38, %c4096 overflow<nsw> : index
              %43 = arith.addi %42, %41 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %26 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%28, %26 : memref<?x?xf32>, memref<?x?xf32>) outs(%30 : memref<?x?xf32>)
              loom.semaphore_give %26 : memref<?x?xf32>
              loom.semaphore_give %28 : memref<?x?xf32>
            }
            %33 = arith.muli %22, %12 : index
            %34 = arith.muli %24, %13 : index
            %35 = arith.muli %33, %c4096 overflow<nsw> : index
            %36 = arith.addi %35, %34 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %30, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %30 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__v_h__BK256__BM128__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %13, %c8 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.muli %12, %c8 overflow<nsw> : index
          %19 = arith.divsi %c4095, %18 : index
          %20 = arith.addi %19, %c1 : index
          scf.for %arg6 = %c0 to %20 step %c1 {
            %21 = arith.muli %arg6, %c8 overflow<nsw> : index
            %22 = arith.addi %arg3, %21 : index
            %23 = arith.muli %arg5, %c8 overflow<nsw> : index
            %24 = arith.addi %arg4, %23 : index
            %25 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %26 = loom.semaphore_take %25 : memref<?x?xf32> -> memref<?x?xf32>
            %27 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %28 = loom.semaphore_take %27 : memref<?x?xf32> -> memref<?x?xf32>
            %29 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%30 : memref<?x?xf32>)
            %31 = arith.divsi %c511, %14 : index
            %32 = arith.addi %31, %c1 : index
            scf.for %arg7 = %c0 to %32 step %c1 {
              %37 = arith.muli %22, %12 : index
              %38 = arith.muli %arg7, %14 : index
              %39 = arith.muli %37, %c512 overflow<nsw> : index
              %40 = arith.addi %39, %38 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%40], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %28 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %41 = arith.muli %24, %13 : index
              %42 = arith.muli %38, %c4096 overflow<nsw> : index
              %43 = arith.addi %42, %41 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %26 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%28, %26 : memref<?x?xf32>, memref<?x?xf32>) outs(%30 : memref<?x?xf32>)
              loom.semaphore_give %26 : memref<?x?xf32>
              loom.semaphore_give %28 : memref<?x?xf32>
            }
            %33 = arith.muli %22, %12 : index
            %34 = arith.muli %24, %13 : index
            %35 = arith.muli %33, %c4096 overflow<nsw> : index
            %36 = arith.addi %35, %34 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %30, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %30 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__d_d__BK32__BM512__BN256(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %12, %c8 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.muli %13, %c8 overflow<nsw> : index
          %19 = arith.divsi %c4095, %18 : index
          %20 = arith.addi %19, %c1 : index
          scf.for %arg6 = %c0 to %20 step %c1 {
            %21 = arith.muli %arg5, %c8 overflow<nsw> : index
            %22 = arith.addi %arg3, %21 : index
            %23 = arith.muli %arg6, %c8 overflow<nsw> : index
            %24 = arith.addi %arg4, %23 : index
            %25 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %26 = loom.semaphore_take %25 : memref<?x?xf32> -> memref<?x?xf32>
            %27 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %28 = loom.semaphore_take %27 : memref<?x?xf32> -> memref<?x?xf32>
            %29 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%30 : memref<?x?xf32>)
            %31 = arith.divsi %c511, %14 : index
            %32 = arith.addi %31, %c1 : index
            scf.for %arg7 = %c0 to %32 step %c1 {
              %37 = arith.muli %22, %12 : index
              %38 = arith.muli %arg7, %14 : index
              %39 = arith.muli %37, %c512 overflow<nsw> : index
              %40 = arith.addi %39, %38 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%40], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %28 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %41 = arith.muli %24, %13 : index
              %42 = arith.muli %38, %c4096 overflow<nsw> : index
              %43 = arith.addi %42, %41 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %26 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%28, %26 : memref<?x?xf32>, memref<?x?xf32>) outs(%30 : memref<?x?xf32>)
              loom.semaphore_give %26 : memref<?x?xf32>
              loom.semaphore_give %28 : memref<?x?xf32>
            }
            %33 = arith.muli %22, %12 : index
            %34 = arith.muli %24, %13 : index
            %35 = arith.muli %33, %c4096 overflow<nsw> : index
            %36 = arith.addi %35, %34 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %30, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %30 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__d_v__BK256__BM256__BN512(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %12, %c8 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.muli %13, %c8 overflow<nsw> : index
          %19 = arith.divsi %c4095, %18 : index
          %20 = arith.addi %19, %c1 : index
          scf.for %arg6 = %c0 to %20 step %c1 {
            %21 = arith.muli %arg5, %c8 overflow<nsw> : index
            %22 = arith.addi %arg3, %21 : index
            %23 = arith.muli %arg6, %c8 overflow<nsw> : index
            %24 = arith.addi %arg4, %23 : index
            %25 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %26 = loom.semaphore_take %25 : memref<?x?xf32> -> memref<?x?xf32>
            %27 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %28 = loom.semaphore_take %27 : memref<?x?xf32> -> memref<?x?xf32>
            %29 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%30 : memref<?x?xf32>)
            %31 = arith.divsi %c511, %14 : index
            %32 = arith.addi %31, %c1 : index
            scf.for %arg7 = %c0 to %32 step %c1 {
              %37 = arith.muli %22, %12 : index
              %38 = arith.muli %arg7, %14 : index
              %39 = arith.muli %37, %c512 overflow<nsw> : index
              %40 = arith.addi %39, %38 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%40], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %28 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %41 = arith.muli %24, %13 : index
              %42 = arith.muli %38, %c4096 overflow<nsw> : index
              %43 = arith.addi %42, %41 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %26 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%28, %26 : memref<?x?xf32>, memref<?x?xf32>) outs(%30 : memref<?x?xf32>)
              loom.semaphore_give %26 : memref<?x?xf32>
              loom.semaphore_give %28 : memref<?x?xf32>
            }
            %33 = arith.muli %22, %12 : index
            %34 = arith.muli %24, %13 : index
            %35 = arith.muli %33, %c4096 overflow<nsw> : index
            %36 = arith.addi %35, %34 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %30, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %30 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__h_d__BK128__BM128__BN512(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %12, %c8 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.muli %13, %c8 overflow<nsw> : index
          %19 = arith.divsi %c4095, %18 : index
          %20 = arith.addi %19, %c1 : index
          scf.for %arg6 = %c0 to %20 step %c1 {
            %21 = arith.muli %arg5, %c8 overflow<nsw> : index
            %22 = arith.addi %arg3, %21 : index
            %23 = arith.muli %arg6, %c8 overflow<nsw> : index
            %24 = arith.addi %arg4, %23 : index
            %25 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %26 = loom.semaphore_take %25 : memref<?x?xf32> -> memref<?x?xf32>
            %27 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %28 = loom.semaphore_take %27 : memref<?x?xf32> -> memref<?x?xf32>
            %29 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%30 : memref<?x?xf32>)
            %31 = arith.divsi %c511, %14 : index
            %32 = arith.addi %31, %c1 : index
            scf.for %arg7 = %c0 to %32 step %c1 {
              %37 = arith.muli %22, %12 : index
              %38 = arith.muli %arg7, %14 : index
              %39 = arith.muli %37, %c512 overflow<nsw> : index
              %40 = arith.addi %39, %38 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%40], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %28 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %41 = arith.muli %24, %13 : index
              %42 = arith.muli %38, %c4096 overflow<nsw> : index
              %43 = arith.addi %42, %41 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %26 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%28, %26 : memref<?x?xf32>, memref<?x?xf32>) outs(%30 : memref<?x?xf32>)
              loom.semaphore_give %26 : memref<?x?xf32>
              loom.semaphore_give %28 : memref<?x?xf32>
            }
            %33 = arith.muli %22, %12 : index
            %34 = arith.muli %24, %13 : index
            %35 = arith.muli %33, %c4096 overflow<nsw> : index
            %36 = arith.addi %35, %34 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %30, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %30 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__h_v__BK64__BM64__BN512(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %12, %c8 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.muli %13, %c8 overflow<nsw> : index
          %19 = arith.divsi %c4095, %18 : index
          %20 = arith.addi %19, %c1 : index
          scf.for %arg6 = %c0 to %20 step %c1 {
            %21 = arith.muli %arg5, %c8 overflow<nsw> : index
            %22 = arith.addi %arg3, %21 : index
            %23 = arith.muli %arg6, %c8 overflow<nsw> : index
            %24 = arith.addi %arg4, %23 : index
            %25 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %26 = loom.semaphore_take %25 : memref<?x?xf32> -> memref<?x?xf32>
            %27 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %28 = loom.semaphore_take %27 : memref<?x?xf32> -> memref<?x?xf32>
            %29 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%30 : memref<?x?xf32>)
            %31 = arith.divsi %c511, %14 : index
            %32 = arith.addi %31, %c1 : index
            scf.for %arg7 = %c0 to %32 step %c1 {
              %37 = arith.muli %22, %12 : index
              %38 = arith.muli %arg7, %14 : index
              %39 = arith.muli %37, %c512 overflow<nsw> : index
              %40 = arith.addi %39, %38 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%40], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %28 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %41 = arith.muli %24, %13 : index
              %42 = arith.muli %38, %c4096 overflow<nsw> : index
              %43 = arith.addi %42, %41 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %26 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%28, %26 : memref<?x?xf32>, memref<?x?xf32>) outs(%30 : memref<?x?xf32>)
              loom.semaphore_give %26 : memref<?x?xf32>
              loom.semaphore_give %28 : memref<?x?xf32>
            }
            %33 = arith.muli %22, %12 : index
            %34 = arith.muli %24, %13 : index
            %35 = arith.muli %33, %c4096 overflow<nsw> : index
            %36 = arith.addi %35, %34 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %30, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %30 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__d_d__BK32__BM512__BN256(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %13, %c8 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.muli %12, %c8 overflow<nsw> : index
          %19 = arith.divsi %c4095, %18 : index
          %20 = arith.addi %19, %c1 : index
          scf.for %arg6 = %c0 to %20 step %c1 {
            %21 = arith.muli %arg6, %c8 overflow<nsw> : index
            %22 = arith.addi %arg3, %21 : index
            %23 = arith.muli %arg5, %c8 overflow<nsw> : index
            %24 = arith.addi %arg4, %23 : index
            %25 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %26 = loom.semaphore_take %25 : memref<?x?xf32> -> memref<?x?xf32>
            %27 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %28 = loom.semaphore_take %27 : memref<?x?xf32> -> memref<?x?xf32>
            %29 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%30 : memref<?x?xf32>)
            %31 = arith.divsi %c511, %14 : index
            %32 = arith.addi %31, %c1 : index
            scf.for %arg7 = %c0 to %32 step %c1 {
              %37 = arith.muli %22, %12 : index
              %38 = arith.muli %arg7, %14 : index
              %39 = arith.muli %37, %c512 overflow<nsw> : index
              %40 = arith.addi %39, %38 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%40], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %28 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %41 = arith.muli %24, %13 : index
              %42 = arith.muli %38, %c4096 overflow<nsw> : index
              %43 = arith.addi %42, %41 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %26 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%28, %26 : memref<?x?xf32>, memref<?x?xf32>) outs(%30 : memref<?x?xf32>)
              loom.semaphore_give %26 : memref<?x?xf32>
              loom.semaphore_give %28 : memref<?x?xf32>
            }
            %33 = arith.muli %22, %12 : index
            %34 = arith.muli %24, %13 : index
            %35 = arith.muli %33, %c4096 overflow<nsw> : index
            %36 = arith.addi %35, %34 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %30, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %30 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__d_v__BK256__BM512__BN128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %13, %c8 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.muli %12, %c8 overflow<nsw> : index
          %19 = arith.divsi %c4095, %18 : index
          %20 = arith.addi %19, %c1 : index
          scf.for %arg6 = %c0 to %20 step %c1 {
            %21 = arith.muli %arg6, %c8 overflow<nsw> : index
            %22 = arith.addi %arg3, %21 : index
            %23 = arith.muli %arg5, %c8 overflow<nsw> : index
            %24 = arith.addi %arg4, %23 : index
            %25 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %26 = loom.semaphore_take %25 : memref<?x?xf32> -> memref<?x?xf32>
            %27 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %28 = loom.semaphore_take %27 : memref<?x?xf32> -> memref<?x?xf32>
            %29 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%30 : memref<?x?xf32>)
            %31 = arith.divsi %c511, %14 : index
            %32 = arith.addi %31, %c1 : index
            scf.for %arg7 = %c0 to %32 step %c1 {
              %37 = arith.muli %22, %12 : index
              %38 = arith.muli %arg7, %14 : index
              %39 = arith.muli %37, %c512 overflow<nsw> : index
              %40 = arith.addi %39, %38 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%40], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %28 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %41 = arith.muli %24, %13 : index
              %42 = arith.muli %38, %c4096 overflow<nsw> : index
              %43 = arith.addi %42, %41 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %26 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%28, %26 : memref<?x?xf32>, memref<?x?xf32>) outs(%30 : memref<?x?xf32>)
              loom.semaphore_give %26 : memref<?x?xf32>
              loom.semaphore_give %28 : memref<?x?xf32>
            }
            %33 = arith.muli %22, %12 : index
            %34 = arith.muli %24, %13 : index
            %35 = arith.muli %33, %c4096 overflow<nsw> : index
            %36 = arith.addi %35, %34 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %30, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %30 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__h_d__BK256__BM256__BN128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %13, %c8 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.muli %12, %c8 overflow<nsw> : index
          %19 = arith.divsi %c4095, %18 : index
          %20 = arith.addi %19, %c1 : index
          scf.for %arg6 = %c0 to %20 step %c1 {
            %21 = arith.muli %arg6, %c8 overflow<nsw> : index
            %22 = arith.addi %arg3, %21 : index
            %23 = arith.muli %arg5, %c8 overflow<nsw> : index
            %24 = arith.addi %arg4, %23 : index
            %25 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %26 = loom.semaphore_take %25 : memref<?x?xf32> -> memref<?x?xf32>
            %27 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %28 = loom.semaphore_take %27 : memref<?x?xf32> -> memref<?x?xf32>
            %29 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%30 : memref<?x?xf32>)
            %31 = arith.divsi %c511, %14 : index
            %32 = arith.addi %31, %c1 : index
            scf.for %arg7 = %c0 to %32 step %c1 {
              %37 = arith.muli %22, %12 : index
              %38 = arith.muli %arg7, %14 : index
              %39 = arith.muli %37, %c512 overflow<nsw> : index
              %40 = arith.addi %39, %38 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%40], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %28 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %41 = arith.muli %24, %13 : index
              %42 = arith.muli %38, %c4096 overflow<nsw> : index
              %43 = arith.addi %42, %41 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %26 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%28, %26 : memref<?x?xf32>, memref<?x?xf32>) outs(%30 : memref<?x?xf32>)
              loom.semaphore_give %26 : memref<?x?xf32>
              loom.semaphore_give %28 : memref<?x?xf32>
            }
            %33 = arith.muli %22, %12 : index
            %34 = arith.muli %24, %13 : index
            %35 = arith.muli %33, %c4096 overflow<nsw> : index
            %36 = arith.addi %35, %34 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %30, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %30 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__h_v__BK64__BM128__BN512(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %13, %c8 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.muli %12, %c8 overflow<nsw> : index
          %19 = arith.divsi %c4095, %18 : index
          %20 = arith.addi %19, %c1 : index
          scf.for %arg6 = %c0 to %20 step %c1 {
            %21 = arith.muli %arg6, %c8 overflow<nsw> : index
            %22 = arith.addi %arg3, %21 : index
            %23 = arith.muli %arg5, %c8 overflow<nsw> : index
            %24 = arith.addi %arg4, %23 : index
            %25 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %26 = loom.semaphore_take %25 : memref<?x?xf32> -> memref<?x?xf32>
            %27 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %28 = loom.semaphore_take %27 : memref<?x?xf32> -> memref<?x?xf32>
            %29 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%30 : memref<?x?xf32>)
            %31 = arith.divsi %c511, %14 : index
            %32 = arith.addi %31, %c1 : index
            scf.for %arg7 = %c0 to %32 step %c1 {
              %37 = arith.muli %22, %12 : index
              %38 = arith.muli %arg7, %14 : index
              %39 = arith.muli %37, %c512 overflow<nsw> : index
              %40 = arith.addi %39, %38 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%40], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %28 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %41 = arith.muli %24, %13 : index
              %42 = arith.muli %38, %c4096 overflow<nsw> : index
              %43 = arith.addi %42, %41 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %26 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%28, %26 : memref<?x?xf32>, memref<?x?xf32>) outs(%30 : memref<?x?xf32>)
              loom.semaphore_give %26 : memref<?x?xf32>
              loom.semaphore_give %28 : memref<?x?xf32>
            }
            %33 = arith.muli %22, %12 : index
            %34 = arith.muli %24, %13 : index
            %35 = arith.muli %33, %c4096 overflow<nsw> : index
            %36 = arith.addi %35, %34 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %30, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %30 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__d_d__BK512__BM4096__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.divsi %c4095, %12 : index
        %16 = arith.addi %15, %c1 : index
        scf.for %arg5 = %c0 to %16 step %c1 {
          %17 = arith.muli %13, %c64 overflow<nsw> : index
          %18 = arith.divsi %c4095, %17 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg6, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %arg5, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %23, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %arg5, %12 : index
            %33 = arith.muli %23, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__a_d__BK512__BM4096__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.divsi %c4095, %12 : index
        %16 = arith.addi %15, %c1 : index
        scf.for %arg5 = %c0 to %16 step %c1 {
          %17 = arith.muli %13, %c64 overflow<nsw> : index
          %18 = arith.divsi %c4095, %17 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg6, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %arg5, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %23, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %arg5, %12 : index
            %33 = arith.muli %23, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__h_d__BK128__BM1024__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.divsi %c4095, %12 : index
        %16 = arith.addi %15, %c1 : index
        scf.for %arg5 = %c0 to %16 step %c1 {
          %17 = arith.muli %13, %c64 overflow<nsw> : index
          %18 = arith.divsi %c4095, %17 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg6, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %arg5, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %23, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %arg5, %12 : index
            %33 = arith.muli %23, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__v_d__BK64__BM2048__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.divsi %c4095, %12 : index
        %16 = arith.addi %15, %c1 : index
        scf.for %arg5 = %c0 to %16 step %c1 {
          %17 = arith.muli %13, %c64 overflow<nsw> : index
          %18 = arith.divsi %c4095, %17 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg6, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %arg5, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %23, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %arg5, %12 : index
            %33 = arith.muli %23, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__d_d__BK512__BM4096__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %13, %c64 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.divsi %c4095, %12 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg5, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %arg6, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %23, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %arg6, %12 : index
            %33 = arith.muli %23, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__a_d__BK512__BM4096__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %13, %c64 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.divsi %c4095, %12 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg5, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %arg6, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %23, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %arg6, %12 : index
            %33 = arith.muli %23, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__h_d__BK32__BM4096__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %13, %c64 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.divsi %c4095, %12 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg5, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %arg6, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %23, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %arg6, %12 : index
            %33 = arith.muli %23, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__v_d__BK128__BM1024__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %13, %c64 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.divsi %c4095, %12 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg5, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %arg6, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %23, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %arg6, %12 : index
            %33 = arith.muli %23, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f01__d_d__BK512__BM4096__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.divsi %c4095, %12 : index
        %16 = arith.addi %15, %c1 : index
        scf.for %arg5 = %c0 to %16 step %c1 {
          %17 = arith.muli %13, %c64 overflow<nsw> : index
          %18 = arith.divsi %c4095, %17 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg6, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %arg5, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %23, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %arg5, %12 : index
            %33 = arith.muli %23, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f01__a_d__BK512__BM4096__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.divsi %c4095, %12 : index
        %16 = arith.addi %15, %c1 : index
        scf.for %arg5 = %c0 to %16 step %c1 {
          %17 = arith.muli %13, %c64 overflow<nsw> : index
          %18 = arith.divsi %c4095, %17 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg6, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %arg5, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %23, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %arg5, %12 : index
            %33 = arith.muli %23, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f01__h_d__BK128__BM1024__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.divsi %c4095, %12 : index
        %16 = arith.addi %15, %c1 : index
        scf.for %arg5 = %c0 to %16 step %c1 {
          %17 = arith.muli %13, %c64 overflow<nsw> : index
          %18 = arith.divsi %c4095, %17 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg6, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %arg5, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %23, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %arg5, %12 : index
            %33 = arith.muli %23, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f01__v_d__BK64__BM2048__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.divsi %c4095, %12 : index
        %16 = arith.addi %15, %c1 : index
        scf.for %arg5 = %c0 to %16 step %c1 {
          %17 = arith.muli %13, %c64 overflow<nsw> : index
          %18 = arith.divsi %c4095, %17 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg6, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %arg5, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %23, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %arg5, %12 : index
            %33 = arith.muli %23, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f10__d_d__BK512__BM4096__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %13, %c64 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.divsi %c4095, %12 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg5, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %arg6, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %23, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %arg6, %12 : index
            %33 = arith.muli %23, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f10__a_d__BK512__BM4096__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %13, %c64 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.divsi %c4095, %12 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg5, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %arg6, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %23, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %arg6, %12 : index
            %33 = arith.muli %23, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f10__h_d__BK32__BM4096__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %13, %c64 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.divsi %c4095, %12 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg5, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %arg6, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %23, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %arg6, %12 : index
            %33 = arith.muli %23, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f10__v_d__BK128__BM1024__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %c511 = arith.constant 511 : index
      %c4095 = arith.constant 4095 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@block_size_0 : index
      %13 = loom.get_symbolic_block_size @constraints::@block_size_1 : index
      %14 = loom.get_symbolic_block_size @constraints::@block_size_2 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %15 = arith.muli %13, %c64 overflow<nsw> : index
        %16 = arith.divsi %c4095, %15 : index
        %17 = arith.addi %16, %c1 : index
        scf.for %arg5 = %c0 to %17 step %c1 {
          %18 = arith.divsi %c4095, %12 : index
          %19 = arith.addi %18, %c1 : index
          scf.for %arg6 = %c0 to %19 step %c1 {
            %20 = arith.muli %arg3, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg4 : index
            %22 = arith.muli %arg5, %c64 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [%14, %13] on @L1 : memref<?x?xf32>
            %25 = loom.semaphore_take %24 : memref<?x?xf32> -> memref<?x?xf32>
            %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf32>
            %27 = loom.semaphore_take %26 : memref<?x?xf32> -> memref<?x?xf32>
            %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
            %29 = loom.semaphore_take %28 : memref<?x?xf32> -> memref<?x?xf32>
            linalg.fill ins(%cst : f32) outs(%29 : memref<?x?xf32>)
            %30 = arith.divsi %c511, %14 : index
            %31 = arith.addi %30, %c1 : index
            scf.for %arg7 = %c0 to %31 step %c1 {
              %36 = arith.muli %arg6, %12 : index
              %37 = arith.muli %arg7, %14 : index
              %38 = arith.muli %36, %c512 overflow<nsw> : index
              %39 = arith.addi %38, %37 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [%12, %14], strides: [512, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32>
              %40 = arith.muli %23, %13 : index
              %41 = arith.muli %37, %c4096 overflow<nsw> : index
              %42 = arith.addi %41, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%42], sizes: [%14, %13], strides: [4096, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %25 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32>
              linalg.matmul ins(%27, %25 : memref<?x?xf32>, memref<?x?xf32>) outs(%29 : memref<?x?xf32>)
              loom.semaphore_give %25 : memref<?x?xf32>
              loom.semaphore_give %27 : memref<?x?xf32>
            }
            %32 = arith.muli %arg6, %12 : index
            %33 = arith.muli %23, %13 : index
            %34 = arith.muli %32, %c4096 overflow<nsw> : index
            %35 = arith.addi %34, %33 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [%12, %13], strides: [4096, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.copy %29, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %29 : memref<?x?xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
}
