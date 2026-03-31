module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
  %0 = adl.memory.bank "DRAM_bank", {bsize = 8192 : i64, nblk = 196608 : i64}
  %1 = adl.spatial_dim "dram_channel", 8
  %2 = adl.memory.array "DRAM", [%1] of %0
  %3 = adl.memory.bank "bank", {bsize = 16 : i64, nblk = 5856 : i64}
  %4 = adl.spatial_dim "nbank", 16
  %5 = adl.memory.array "L1", [%4] of %3
  %6 = adl.processor.compute @matrix_lane, [(%5, %5)]
  %7 = adl.processor.compute @vector_lane, [(%5, %5)]
  %8 = adl.arch.compose "core", arch[%6, %7], mem[%5]
  %9 = adl.spatial_dim "x", 8
  %10 = adl.spatial_dim "y", 8
  %11 = adl.arch.scale "mesh", [%9, %10] of %8
  %12 = adl.processor.dmover @dram_l1_mover, [(%2, %5), (%5, %2)]
  %13 = adl.arch.compose "system", arch[%11, %12], mem[%2]
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f01__n_n_n__block_size_064__block_size_14096__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %17 = loom.semaphore_take %16 : memref<64x4096xf16> -> memref<64x4096xf16>
        %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
        %20 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %21 = loom.semaphore_take %20 : memref<64x4096xf16> -> memref<64x4096xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %24 = arith.muli %15, %c32768 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>> to memref<64x64xf16>
          %26 = arith.muli %arg5, %c262144 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x4096xf16, strided<[4096, 1], offset: ?>> to memref<64x4096xf16>
          loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x4096xf16>) outs(%21 : memref<64x4096xf16>)
          loom.semaphore_give %17 : memref<64x4096xf16>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        %cast = memref.cast %21 : memref<64x4096xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f01__n_y_n__block_size_064__block_size_14096__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %17 = loom.semaphore_take %16 : memref<64x4096xf16> -> memref<64x4096xf16>
        %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
        %20 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %21 = loom.semaphore_take %20 : memref<64x4096xf16> -> memref<64x4096xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %24 = arith.muli %15, %c32768 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>> to memref<64x64xf16>
          %26 = arith.muli %arg5, %c262144 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<64x4096xf16, strided<[4096, 1], offset: ?>> to memref<64x4096xf16>
          loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x4096xf16>) outs(%21 : memref<64x4096xf16>)
          loom.semaphore_give %17 : memref<64x4096xf16>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        %cast = memref.cast %21 : memref<64x4096xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f01__n_x_n__block_size_064__block_size_14096__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %17 = loom.semaphore_take %16 : memref<64x4096xf16> -> memref<64x4096xf16>
        %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
        %20 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %21 = loom.semaphore_take %20 : memref<64x4096xf16> -> memref<64x4096xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %24 = arith.muli %15, %c32768 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>> to memref<64x64xf16>
          %26 = arith.muli %arg5, %c262144 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<64x4096xf16, strided<[4096, 1], offset: ?>> to memref<64x4096xf16>
          loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x4096xf16>) outs(%21 : memref<64x4096xf16>)
          loom.semaphore_give %17 : memref<64x4096xf16>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        %cast = memref.cast %21 : memref<64x4096xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f01__n_a_n__block_size_064__block_size_12048__block_size_2256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c1048576 = arith.constant 1048576 : index
      %c32768 = arith.constant 32768 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c2048 = arith.constant 2048 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [256, 2048] on @L1 : memref<256x2048xf16>
          %17 = loom.semaphore_take %16 : memref<256x2048xf16> -> memref<256x2048xf16>
          %18 = loom.alloc [64, 256] on @L1 : memref<64x256xf16>
          %19 = loom.semaphore_take %18 : memref<64x256xf16> -> memref<64x256xf16>
          %20 = loom.alloc [64, 2048] on @L1 : memref<64x2048xf16>
          %21 = loom.semaphore_take %20 : memref<64x2048xf16> -> memref<64x2048xf16>
          scf.for %arg6 = %c0 to %c2 step %c1 {
            %25 = arith.muli %arg6, %c256 : index
            %26 = arith.muli %15, %c32768 : index
            %27 = arith.addi %26, %25 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 256], strides: [512, 1] : memref<4096x512xf16> to memref<64x256xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x256xf16, strided<[512, 1], offset: ?>> to memref<64x256xf16>
            %28 = arith.muli %arg5, %c2048 : index
            %29 = arith.muli %arg6, %c1048576 : index
            %30 = arith.addi %29, %28 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [256, 2048], strides: [4096, 1] : memref<512x4096xf16> to memref<256x2048xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<256x2048xf16, strided<[4096, 1], offset: ?>> to memref<256x2048xf16>
            loom.matmul ins(%19, %17 : memref<64x256xf16>, memref<256x2048xf16>) outs(%21 : memref<64x2048xf16>)
            loom.semaphore_give %17 : memref<256x2048xf16>
            loom.semaphore_give %19 : memref<64x256xf16>
          }
          %cast = memref.cast %21 : memref<64x2048xf16> to memref<?x?xf16>
          %22 = arith.muli %arg5, %c2048 : index
          %23 = arith.muli %15, %c262144 : index
          %24 = arith.addi %23, %22 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 2048], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x2048xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<64x2048xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<64x2048xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f10__n_n_n__block_size_064__block_size_14096__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %17 = loom.semaphore_take %16 : memref<64x4096xf16> -> memref<64x4096xf16>
        %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
        %20 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %21 = loom.semaphore_take %20 : memref<64x4096xf16> -> memref<64x4096xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %24 = arith.muli %15, %c32768 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>> to memref<64x64xf16>
          %26 = arith.muli %arg5, %c262144 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x4096xf16, strided<[4096, 1], offset: ?>> to memref<64x4096xf16>
          loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x4096xf16>) outs(%21 : memref<64x4096xf16>)
          loom.semaphore_give %17 : memref<64x4096xf16>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        %cast = memref.cast %21 : memref<64x4096xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f10__n_y_n__block_size_064__block_size_14096__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %17 = loom.semaphore_take %16 : memref<64x4096xf16> -> memref<64x4096xf16>
        %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
        %20 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %21 = loom.semaphore_take %20 : memref<64x4096xf16> -> memref<64x4096xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %24 = arith.muli %15, %c32768 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>> to memref<64x64xf16>
          %26 = arith.muli %arg5, %c262144 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<64x4096xf16, strided<[4096, 1], offset: ?>> to memref<64x4096xf16>
          loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x4096xf16>) outs(%21 : memref<64x4096xf16>)
          loom.semaphore_give %17 : memref<64x4096xf16>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        %cast = memref.cast %21 : memref<64x4096xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f10__n_x_n__block_size_064__block_size_14096__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %17 = loom.semaphore_take %16 : memref<64x4096xf16> -> memref<64x4096xf16>
        %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
        %20 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %21 = loom.semaphore_take %20 : memref<64x4096xf16> -> memref<64x4096xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %24 = arith.muli %15, %c32768 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>> to memref<64x64xf16>
          %26 = arith.muli %arg5, %c262144 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<64x4096xf16, strided<[4096, 1], offset: ?>> to memref<64x4096xf16>
          loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x4096xf16>) outs(%21 : memref<64x4096xf16>)
          loom.semaphore_give %17 : memref<64x4096xf16>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        %cast = memref.cast %21 : memref<64x4096xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f10__n_a_n__block_size_064__block_size_11024__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c1024 = arith.constant 1024 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [512, 1024] on @L1 : memref<512x1024xf16>
          %17 = loom.semaphore_take %16 : memref<512x1024xf16> -> memref<512x1024xf16>
          %18 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
          %19 = loom.semaphore_take %18 : memref<64x512xf16> -> memref<64x512xf16>
          %20 = loom.alloc [64, 1024] on @L1 : memref<64x1024xf16>
          %21 = loom.semaphore_take %20 : memref<64x1024xf16> -> memref<64x1024xf16>
          %22 = arith.muli %15, %c32768 : index
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
          %23 = arith.muli %arg5, %c1024 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%23], sizes: [512, 1024], strides: [4096, 1] : memref<512x4096xf16> to memref<512x1024xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<512x1024xf16, strided<[4096, 1], offset: ?>> to memref<512x1024xf16>
          loom.matmul ins(%19, %17 : memref<64x512xf16>, memref<512x1024xf16>) outs(%21 : memref<64x1024xf16>)
          loom.semaphore_give %17 : memref<512x1024xf16>
          loom.semaphore_give %19 : memref<64x512xf16>
          %cast = memref.cast %21 : memref<64x1024xf16> to memref<?x?xf16>
          %24 = arith.muli %15, %c262144 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [64, 1024], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x1024xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<64x1024xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<64x1024xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f01__n_n_n__block_size_064__block_size_14096__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %17 = loom.semaphore_take %16 : memref<64x4096xf16> -> memref<64x4096xf16>
        %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
        %20 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %21 = loom.semaphore_take %20 : memref<64x4096xf16> -> memref<64x4096xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %24 = arith.muli %15, %c32768 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>> to memref<64x64xf16>
          %26 = arith.muli %arg5, %c262144 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x4096xf16, strided<[4096, 1], offset: ?>> to memref<64x4096xf16>
          loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x4096xf16>) outs(%21 : memref<64x4096xf16>)
          loom.semaphore_give %17 : memref<64x4096xf16>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        %cast = memref.cast %21 : memref<64x4096xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f01__n_y_n__block_size_064__block_size_14096__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %17 = loom.semaphore_take %16 : memref<64x4096xf16> -> memref<64x4096xf16>
        %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
        %20 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %21 = loom.semaphore_take %20 : memref<64x4096xf16> -> memref<64x4096xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %24 = arith.muli %15, %c32768 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>> to memref<64x64xf16>
          %26 = arith.muli %arg5, %c262144 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<64x4096xf16, strided<[4096, 1], offset: ?>> to memref<64x4096xf16>
          loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x4096xf16>) outs(%21 : memref<64x4096xf16>)
          loom.semaphore_give %17 : memref<64x4096xf16>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        %cast = memref.cast %21 : memref<64x4096xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f01__n_x_n__block_size_064__block_size_14096__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %17 = loom.semaphore_take %16 : memref<64x4096xf16> -> memref<64x4096xf16>
        %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
        %20 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %21 = loom.semaphore_take %20 : memref<64x4096xf16> -> memref<64x4096xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %24 = arith.muli %15, %c32768 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>> to memref<64x64xf16>
          %26 = arith.muli %arg5, %c262144 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<64x4096xf16, strided<[4096, 1], offset: ?>> to memref<64x4096xf16>
          loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x4096xf16>) outs(%21 : memref<64x4096xf16>)
          loom.semaphore_give %17 : memref<64x4096xf16>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        %cast = memref.cast %21 : memref<64x4096xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f01__n_a_n__block_size_064__block_size_12048__block_size_2256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c1048576 = arith.constant 1048576 : index
      %c32768 = arith.constant 32768 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c2048 = arith.constant 2048 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [256, 2048] on @L1 : memref<256x2048xf16>
          %17 = loom.semaphore_take %16 : memref<256x2048xf16> -> memref<256x2048xf16>
          %18 = loom.alloc [64, 256] on @L1 : memref<64x256xf16>
          %19 = loom.semaphore_take %18 : memref<64x256xf16> -> memref<64x256xf16>
          %20 = loom.alloc [64, 2048] on @L1 : memref<64x2048xf16>
          %21 = loom.semaphore_take %20 : memref<64x2048xf16> -> memref<64x2048xf16>
          scf.for %arg6 = %c0 to %c2 step %c1 {
            %25 = arith.muli %arg6, %c256 : index
            %26 = arith.muli %15, %c32768 : index
            %27 = arith.addi %26, %25 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 256], strides: [512, 1] : memref<4096x512xf16> to memref<64x256xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x256xf16, strided<[512, 1], offset: ?>> to memref<64x256xf16>
            %28 = arith.muli %arg5, %c2048 : index
            %29 = arith.muli %arg6, %c1048576 : index
            %30 = arith.addi %29, %28 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [256, 2048], strides: [4096, 1] : memref<512x4096xf16> to memref<256x2048xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<256x2048xf16, strided<[4096, 1], offset: ?>> to memref<256x2048xf16>
            loom.matmul ins(%19, %17 : memref<64x256xf16>, memref<256x2048xf16>) outs(%21 : memref<64x2048xf16>)
            loom.semaphore_give %17 : memref<256x2048xf16>
            loom.semaphore_give %19 : memref<64x256xf16>
          }
          %cast = memref.cast %21 : memref<64x2048xf16> to memref<?x?xf16>
          %22 = arith.muli %arg5, %c2048 : index
          %23 = arith.muli %15, %c262144 : index
          %24 = arith.addi %23, %22 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 2048], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x2048xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<64x2048xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<64x2048xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f10__n_n_n__block_size_064__block_size_14096__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %17 = loom.semaphore_take %16 : memref<64x4096xf16> -> memref<64x4096xf16>
        %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
        %20 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %21 = loom.semaphore_take %20 : memref<64x4096xf16> -> memref<64x4096xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %24 = arith.muli %15, %c32768 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>> to memref<64x64xf16>
          %26 = arith.muli %arg5, %c262144 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x4096xf16, strided<[4096, 1], offset: ?>> to memref<64x4096xf16>
          loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x4096xf16>) outs(%21 : memref<64x4096xf16>)
          loom.semaphore_give %17 : memref<64x4096xf16>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        %cast = memref.cast %21 : memref<64x4096xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f10__n_y_n__block_size_064__block_size_14096__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %17 = loom.semaphore_take %16 : memref<64x4096xf16> -> memref<64x4096xf16>
        %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
        %20 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %21 = loom.semaphore_take %20 : memref<64x4096xf16> -> memref<64x4096xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %24 = arith.muli %15, %c32768 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>> to memref<64x64xf16>
          %26 = arith.muli %arg5, %c262144 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<64x4096xf16, strided<[4096, 1], offset: ?>> to memref<64x4096xf16>
          loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x4096xf16>) outs(%21 : memref<64x4096xf16>)
          loom.semaphore_give %17 : memref<64x4096xf16>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        %cast = memref.cast %21 : memref<64x4096xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f10__n_x_n__block_size_064__block_size_14096__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %17 = loom.semaphore_take %16 : memref<64x4096xf16> -> memref<64x4096xf16>
        %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
        %20 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %21 = loom.semaphore_take %20 : memref<64x4096xf16> -> memref<64x4096xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %24 = arith.muli %15, %c32768 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>> to memref<64x64xf16>
          %26 = arith.muli %arg5, %c262144 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<64x4096xf16, strided<[4096, 1], offset: ?>> to memref<64x4096xf16>
          loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x4096xf16>) outs(%21 : memref<64x4096xf16>)
          loom.semaphore_give %17 : memref<64x4096xf16>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        %cast = memref.cast %21 : memref<64x4096xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f10__n_a_n__block_size_064__block_size_11024__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c1024 = arith.constant 1024 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [512, 1024] on @L1 : memref<512x1024xf16>
          %17 = loom.semaphore_take %16 : memref<512x1024xf16> -> memref<512x1024xf16>
          %18 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
          %19 = loom.semaphore_take %18 : memref<64x512xf16> -> memref<64x512xf16>
          %20 = loom.alloc [64, 1024] on @L1 : memref<64x1024xf16>
          %21 = loom.semaphore_take %20 : memref<64x1024xf16> -> memref<64x1024xf16>
          %22 = arith.muli %15, %c32768 : index
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
          %23 = arith.muli %arg5, %c1024 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%23], sizes: [512, 1024], strides: [4096, 1] : memref<512x4096xf16> to memref<512x1024xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<512x1024xf16, strided<[4096, 1], offset: ?>> to memref<512x1024xf16>
          loom.matmul ins(%19, %17 : memref<64x512xf16>, memref<512x1024xf16>) outs(%21 : memref<64x1024xf16>)
          loom.semaphore_give %17 : memref<512x1024xf16>
          loom.semaphore_give %19 : memref<64x512xf16>
          %cast = memref.cast %21 : memref<64x1024xf16> to memref<?x?xf16>
          %24 = arith.muli %15, %c262144 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [64, 1024], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x1024xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<64x1024xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<64x1024xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f01__n_n_n__block_size_0512__block_size_1512__block_size_2256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
        %15 = loom.semaphore_take %14 : memref<256x512xf16> -> memref<256x512xf16>
        %16 = loom.alloc [512, 256] on @L1 : memref<512x256xf16>
        %17 = loom.semaphore_take %16 : memref<512x256xf16> -> memref<512x256xf16>
        %18 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %19 = loom.semaphore_take %18 : memref<512x512xf16> -> memref<512x512xf16>
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %23 = arith.muli %arg5, %c256 : index
          %24 = arith.muli %arg3, %c262144 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [512, 256], strides: [512, 1] : memref<4096x512xf16> to memref<512x256xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<512x256xf16, strided<[512, 1], offset: ?>> to memref<512x256xf16>
          %26 = arith.muli %arg4, %c512 : index
          %27 = arith.muli %arg5, %c1048576 : index
          %28 = arith.addi %27, %26 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [256, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<256x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<256x512xf16, strided<[4096, 1], offset: ?>> to memref<256x512xf16>
          loom.matmul ins(%17, %15 : memref<512x256xf16>, memref<256x512xf16>) outs(%19 : memref<512x512xf16>)
          loom.semaphore_give %15 : memref<256x512xf16>
          loom.semaphore_give %17 : memref<512x256xf16>
        }
        %cast = memref.cast %19 : memref<512x512xf16> to memref<?x?xf16>
        %20 = arith.muli %arg4, %c512 : index
        %21 = arith.muli %arg3, %c2097152 : index
        %22 = arith.addi %21, %20 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f01__n_y_n__block_size_0512__block_size_1512__block_size_2256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
        %15 = loom.semaphore_take %14 : memref<256x512xf16> -> memref<256x512xf16>
        %16 = loom.alloc [512, 256] on @L1 : memref<512x256xf16>
        %17 = loom.semaphore_take %16 : memref<512x256xf16> -> memref<512x256xf16>
        %18 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %19 = loom.semaphore_take %18 : memref<512x512xf16> -> memref<512x512xf16>
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %23 = arith.muli %arg5, %c256 : index
          %24 = arith.muli %arg3, %c262144 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [512, 256], strides: [512, 1] : memref<4096x512xf16> to memref<512x256xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<512x256xf16, strided<[512, 1], offset: ?>> to memref<512x256xf16>
          %26 = arith.muli %arg4, %c512 : index
          %27 = arith.muli %arg5, %c1048576 : index
          %28 = arith.addi %27, %26 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [256, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<256x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<256x512xf16, strided<[4096, 1], offset: ?>> to memref<256x512xf16>
          loom.matmul ins(%17, %15 : memref<512x256xf16>, memref<256x512xf16>) outs(%19 : memref<512x512xf16>)
          loom.semaphore_give %15 : memref<256x512xf16>
          loom.semaphore_give %17 : memref<512x256xf16>
        }
        %cast = memref.cast %19 : memref<512x512xf16> to memref<?x?xf16>
        %20 = arith.muli %arg4, %c512 : index
        %21 = arith.muli %arg3, %c2097152 : index
        %22 = arith.addi %21, %20 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f01__x_n_n__block_size_0512__block_size_1512__block_size_2256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
        %15 = loom.semaphore_take %14 : memref<256x512xf16> -> memref<256x512xf16>
        %16 = loom.alloc [512, 256] on @L1 : memref<512x256xf16>
        %17 = loom.semaphore_take %16 : memref<512x256xf16> -> memref<512x256xf16>
        %18 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %19 = loom.semaphore_take %18 : memref<512x512xf16> -> memref<512x512xf16>
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %23 = arith.muli %arg5, %c256 : index
          %24 = arith.muli %arg3, %c262144 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [512, 256], strides: [512, 1] : memref<4096x512xf16> to memref<512x256xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<512x256xf16, strided<[512, 1], offset: ?>> to memref<512x256xf16>
          %26 = arith.muli %arg4, %c512 : index
          %27 = arith.muli %arg5, %c1048576 : index
          %28 = arith.addi %27, %26 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [256, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<256x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<256x512xf16, strided<[4096, 1], offset: ?>> to memref<256x512xf16>
          loom.matmul ins(%17, %15 : memref<512x256xf16>, memref<256x512xf16>) outs(%19 : memref<512x512xf16>)
          loom.semaphore_give %15 : memref<256x512xf16>
          loom.semaphore_give %17 : memref<512x256xf16>
        }
        %cast = memref.cast %19 : memref<512x512xf16> to memref<?x?xf16>
        %20 = arith.muli %arg4, %c512 : index
        %21 = arith.muli %arg3, %c2097152 : index
        %22 = arith.addi %21, %20 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f01__x_y_n__block_size_0512__block_size_1512__block_size_2256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
        %15 = loom.semaphore_take %14 : memref<256x512xf16> -> memref<256x512xf16>
        %16 = loom.alloc [512, 256] on @L1 : memref<512x256xf16>
        %17 = loom.semaphore_take %16 : memref<512x256xf16> -> memref<512x256xf16>
        %18 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %19 = loom.semaphore_take %18 : memref<512x512xf16> -> memref<512x512xf16>
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %23 = arith.muli %arg5, %c256 : index
          %24 = arith.muli %arg3, %c262144 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [512, 256], strides: [512, 1] : memref<4096x512xf16> to memref<512x256xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<512x256xf16, strided<[512, 1], offset: ?>> to memref<512x256xf16>
          %26 = arith.muli %arg4, %c512 : index
          %27 = arith.muli %arg5, %c1048576 : index
          %28 = arith.addi %27, %26 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [256, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<256x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<256x512xf16, strided<[4096, 1], offset: ?>> to memref<256x512xf16>
          loom.matmul ins(%17, %15 : memref<512x256xf16>, memref<256x512xf16>) outs(%19 : memref<512x512xf16>)
          loom.semaphore_give %15 : memref<256x512xf16>
          loom.semaphore_give %17 : memref<512x256xf16>
        }
        %cast = memref.cast %19 : memref<512x512xf16> to memref<?x?xf16>
        %20 = arith.muli %arg4, %c512 : index
        %21 = arith.muli %arg3, %c2097152 : index
        %22 = arith.addi %21, %20 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f10__n_n_n__block_size_0512__block_size_1512__block_size_2256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
        %15 = loom.semaphore_take %14 : memref<256x512xf16> -> memref<256x512xf16>
        %16 = loom.alloc [512, 256] on @L1 : memref<512x256xf16>
        %17 = loom.semaphore_take %16 : memref<512x256xf16> -> memref<512x256xf16>
        %18 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %19 = loom.semaphore_take %18 : memref<512x512xf16> -> memref<512x512xf16>
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %23 = arith.muli %arg5, %c256 : index
          %24 = arith.muli %arg3, %c262144 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [512, 256], strides: [512, 1] : memref<4096x512xf16> to memref<512x256xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<512x256xf16, strided<[512, 1], offset: ?>> to memref<512x256xf16>
          %26 = arith.muli %arg4, %c512 : index
          %27 = arith.muli %arg5, %c1048576 : index
          %28 = arith.addi %27, %26 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [256, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<256x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<256x512xf16, strided<[4096, 1], offset: ?>> to memref<256x512xf16>
          loom.matmul ins(%17, %15 : memref<512x256xf16>, memref<256x512xf16>) outs(%19 : memref<512x512xf16>)
          loom.semaphore_give %15 : memref<256x512xf16>
          loom.semaphore_give %17 : memref<512x256xf16>
        }
        %cast = memref.cast %19 : memref<512x512xf16> to memref<?x?xf16>
        %20 = arith.muli %arg4, %c512 : index
        %21 = arith.muli %arg3, %c2097152 : index
        %22 = arith.addi %21, %20 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f10__n_y_n__block_size_0512__block_size_1512__block_size_2256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
        %15 = loom.semaphore_take %14 : memref<256x512xf16> -> memref<256x512xf16>
        %16 = loom.alloc [512, 256] on @L1 : memref<512x256xf16>
        %17 = loom.semaphore_take %16 : memref<512x256xf16> -> memref<512x256xf16>
        %18 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %19 = loom.semaphore_take %18 : memref<512x512xf16> -> memref<512x512xf16>
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %23 = arith.muli %arg5, %c256 : index
          %24 = arith.muli %arg3, %c262144 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [512, 256], strides: [512, 1] : memref<4096x512xf16> to memref<512x256xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<512x256xf16, strided<[512, 1], offset: ?>> to memref<512x256xf16>
          %26 = arith.muli %arg4, %c512 : index
          %27 = arith.muli %arg5, %c1048576 : index
          %28 = arith.addi %27, %26 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [256, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<256x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<256x512xf16, strided<[4096, 1], offset: ?>> to memref<256x512xf16>
          loom.matmul ins(%17, %15 : memref<512x256xf16>, memref<256x512xf16>) outs(%19 : memref<512x512xf16>)
          loom.semaphore_give %15 : memref<256x512xf16>
          loom.semaphore_give %17 : memref<512x256xf16>
        }
        %cast = memref.cast %19 : memref<512x512xf16> to memref<?x?xf16>
        %20 = arith.muli %arg4, %c512 : index
        %21 = arith.muli %arg3, %c2097152 : index
        %22 = arith.addi %21, %20 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f10__x_n_n__block_size_0512__block_size_1512__block_size_2256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
        %15 = loom.semaphore_take %14 : memref<256x512xf16> -> memref<256x512xf16>
        %16 = loom.alloc [512, 256] on @L1 : memref<512x256xf16>
        %17 = loom.semaphore_take %16 : memref<512x256xf16> -> memref<512x256xf16>
        %18 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %19 = loom.semaphore_take %18 : memref<512x512xf16> -> memref<512x512xf16>
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %23 = arith.muli %arg5, %c256 : index
          %24 = arith.muli %arg3, %c262144 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [512, 256], strides: [512, 1] : memref<4096x512xf16> to memref<512x256xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<512x256xf16, strided<[512, 1], offset: ?>> to memref<512x256xf16>
          %26 = arith.muli %arg4, %c512 : index
          %27 = arith.muli %arg5, %c1048576 : index
          %28 = arith.addi %27, %26 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [256, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<256x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<256x512xf16, strided<[4096, 1], offset: ?>> to memref<256x512xf16>
          loom.matmul ins(%17, %15 : memref<512x256xf16>, memref<256x512xf16>) outs(%19 : memref<512x512xf16>)
          loom.semaphore_give %15 : memref<256x512xf16>
          loom.semaphore_give %17 : memref<512x256xf16>
        }
        %cast = memref.cast %19 : memref<512x512xf16> to memref<?x?xf16>
        %20 = arith.muli %arg4, %c512 : index
        %21 = arith.muli %arg3, %c2097152 : index
        %22 = arith.addi %21, %20 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f10__x_y_n__block_size_0512__block_size_1512__block_size_2256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
        %15 = loom.semaphore_take %14 : memref<256x512xf16> -> memref<256x512xf16>
        %16 = loom.alloc [512, 256] on @L1 : memref<512x256xf16>
        %17 = loom.semaphore_take %16 : memref<512x256xf16> -> memref<512x256xf16>
        %18 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %19 = loom.semaphore_take %18 : memref<512x512xf16> -> memref<512x512xf16>
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %23 = arith.muli %arg5, %c256 : index
          %24 = arith.muli %arg3, %c262144 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [512, 256], strides: [512, 1] : memref<4096x512xf16> to memref<512x256xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<512x256xf16, strided<[512, 1], offset: ?>> to memref<512x256xf16>
          %26 = arith.muli %arg4, %c512 : index
          %27 = arith.muli %arg5, %c1048576 : index
          %28 = arith.addi %27, %26 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [256, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<256x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<256x512xf16, strided<[4096, 1], offset: ?>> to memref<256x512xf16>
          loom.matmul ins(%17, %15 : memref<512x256xf16>, memref<256x512xf16>) outs(%19 : memref<512x512xf16>)
          loom.semaphore_give %15 : memref<256x512xf16>
          loom.semaphore_give %17 : memref<512x256xf16>
        }
        %cast = memref.cast %19 : memref<512x512xf16> to memref<?x?xf16>
        %20 = arith.muli %arg4, %c512 : index
        %21 = arith.muli %arg3, %c2097152 : index
        %22 = arith.addi %21, %20 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f01__n_n_n__block_size_0512__block_size_1512__block_size_2256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
        %15 = loom.semaphore_take %14 : memref<256x512xf16> -> memref<256x512xf16>
        %16 = loom.alloc [512, 256] on @L1 : memref<512x256xf16>
        %17 = loom.semaphore_take %16 : memref<512x256xf16> -> memref<512x256xf16>
        %18 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %19 = loom.semaphore_take %18 : memref<512x512xf16> -> memref<512x512xf16>
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %23 = arith.muli %arg5, %c256 : index
          %24 = arith.muli %arg3, %c262144 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [512, 256], strides: [512, 1] : memref<4096x512xf16> to memref<512x256xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<512x256xf16, strided<[512, 1], offset: ?>> to memref<512x256xf16>
          %26 = arith.muli %arg4, %c512 : index
          %27 = arith.muli %arg5, %c1048576 : index
          %28 = arith.addi %27, %26 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [256, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<256x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<256x512xf16, strided<[4096, 1], offset: ?>> to memref<256x512xf16>
          loom.matmul ins(%17, %15 : memref<512x256xf16>, memref<256x512xf16>) outs(%19 : memref<512x512xf16>)
          loom.semaphore_give %15 : memref<256x512xf16>
          loom.semaphore_give %17 : memref<512x256xf16>
        }
        %cast = memref.cast %19 : memref<512x512xf16> to memref<?x?xf16>
        %20 = arith.muli %arg4, %c512 : index
        %21 = arith.muli %arg3, %c2097152 : index
        %22 = arith.addi %21, %20 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f01__n_x_n__block_size_0512__block_size_1512__block_size_2256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
        %15 = loom.semaphore_take %14 : memref<256x512xf16> -> memref<256x512xf16>
        %16 = loom.alloc [512, 256] on @L1 : memref<512x256xf16>
        %17 = loom.semaphore_take %16 : memref<512x256xf16> -> memref<512x256xf16>
        %18 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %19 = loom.semaphore_take %18 : memref<512x512xf16> -> memref<512x512xf16>
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %23 = arith.muli %arg5, %c256 : index
          %24 = arith.muli %arg3, %c262144 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [512, 256], strides: [512, 1] : memref<4096x512xf16> to memref<512x256xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<512x256xf16, strided<[512, 1], offset: ?>> to memref<512x256xf16>
          %26 = arith.muli %arg4, %c512 : index
          %27 = arith.muli %arg5, %c1048576 : index
          %28 = arith.addi %27, %26 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [256, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<256x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<256x512xf16, strided<[4096, 1], offset: ?>> to memref<256x512xf16>
          loom.matmul ins(%17, %15 : memref<512x256xf16>, memref<256x512xf16>) outs(%19 : memref<512x512xf16>)
          loom.semaphore_give %15 : memref<256x512xf16>
          loom.semaphore_give %17 : memref<512x256xf16>
        }
        %cast = memref.cast %19 : memref<512x512xf16> to memref<?x?xf16>
        %20 = arith.muli %arg4, %c512 : index
        %21 = arith.muli %arg3, %c2097152 : index
        %22 = arith.addi %21, %20 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f01__y_n_n__block_size_0512__block_size_1512__block_size_2256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
        %15 = loom.semaphore_take %14 : memref<256x512xf16> -> memref<256x512xf16>
        %16 = loom.alloc [512, 256] on @L1 : memref<512x256xf16>
        %17 = loom.semaphore_take %16 : memref<512x256xf16> -> memref<512x256xf16>
        %18 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %19 = loom.semaphore_take %18 : memref<512x512xf16> -> memref<512x512xf16>
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %23 = arith.muli %arg5, %c256 : index
          %24 = arith.muli %arg3, %c262144 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [512, 256], strides: [512, 1] : memref<4096x512xf16> to memref<512x256xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<512x256xf16, strided<[512, 1], offset: ?>> to memref<512x256xf16>
          %26 = arith.muli %arg4, %c512 : index
          %27 = arith.muli %arg5, %c1048576 : index
          %28 = arith.addi %27, %26 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [256, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<256x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<256x512xf16, strided<[4096, 1], offset: ?>> to memref<256x512xf16>
          loom.matmul ins(%17, %15 : memref<512x256xf16>, memref<256x512xf16>) outs(%19 : memref<512x512xf16>)
          loom.semaphore_give %15 : memref<256x512xf16>
          loom.semaphore_give %17 : memref<512x256xf16>
        }
        %cast = memref.cast %19 : memref<512x512xf16> to memref<?x?xf16>
        %20 = arith.muli %arg4, %c512 : index
        %21 = arith.muli %arg3, %c2097152 : index
        %22 = arith.addi %21, %20 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f01__y_x_n__block_size_0512__block_size_1512__block_size_2256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
        %15 = loom.semaphore_take %14 : memref<256x512xf16> -> memref<256x512xf16>
        %16 = loom.alloc [512, 256] on @L1 : memref<512x256xf16>
        %17 = loom.semaphore_take %16 : memref<512x256xf16> -> memref<512x256xf16>
        %18 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %19 = loom.semaphore_take %18 : memref<512x512xf16> -> memref<512x512xf16>
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %23 = arith.muli %arg5, %c256 : index
          %24 = arith.muli %arg3, %c262144 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [512, 256], strides: [512, 1] : memref<4096x512xf16> to memref<512x256xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<512x256xf16, strided<[512, 1], offset: ?>> to memref<512x256xf16>
          %26 = arith.muli %arg4, %c512 : index
          %27 = arith.muli %arg5, %c1048576 : index
          %28 = arith.addi %27, %26 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [256, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<256x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<256x512xf16, strided<[4096, 1], offset: ?>> to memref<256x512xf16>
          loom.matmul ins(%17, %15 : memref<512x256xf16>, memref<256x512xf16>) outs(%19 : memref<512x512xf16>)
          loom.semaphore_give %15 : memref<256x512xf16>
          loom.semaphore_give %17 : memref<512x256xf16>
        }
        %cast = memref.cast %19 : memref<512x512xf16> to memref<?x?xf16>
        %20 = arith.muli %arg4, %c512 : index
        %21 = arith.muli %arg3, %c2097152 : index
        %22 = arith.addi %21, %20 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f10__n_n_n__block_size_0512__block_size_1512__block_size_2256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
        %15 = loom.semaphore_take %14 : memref<256x512xf16> -> memref<256x512xf16>
        %16 = loom.alloc [512, 256] on @L1 : memref<512x256xf16>
        %17 = loom.semaphore_take %16 : memref<512x256xf16> -> memref<512x256xf16>
        %18 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %19 = loom.semaphore_take %18 : memref<512x512xf16> -> memref<512x512xf16>
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %23 = arith.muli %arg5, %c256 : index
          %24 = arith.muli %arg3, %c262144 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [512, 256], strides: [512, 1] : memref<4096x512xf16> to memref<512x256xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<512x256xf16, strided<[512, 1], offset: ?>> to memref<512x256xf16>
          %26 = arith.muli %arg4, %c512 : index
          %27 = arith.muli %arg5, %c1048576 : index
          %28 = arith.addi %27, %26 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [256, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<256x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<256x512xf16, strided<[4096, 1], offset: ?>> to memref<256x512xf16>
          loom.matmul ins(%17, %15 : memref<512x256xf16>, memref<256x512xf16>) outs(%19 : memref<512x512xf16>)
          loom.semaphore_give %15 : memref<256x512xf16>
          loom.semaphore_give %17 : memref<512x256xf16>
        }
        %cast = memref.cast %19 : memref<512x512xf16> to memref<?x?xf16>
        %20 = arith.muli %arg4, %c512 : index
        %21 = arith.muli %arg3, %c2097152 : index
        %22 = arith.addi %21, %20 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f10__n_x_n__block_size_0512__block_size_1512__block_size_2256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
        %15 = loom.semaphore_take %14 : memref<256x512xf16> -> memref<256x512xf16>
        %16 = loom.alloc [512, 256] on @L1 : memref<512x256xf16>
        %17 = loom.semaphore_take %16 : memref<512x256xf16> -> memref<512x256xf16>
        %18 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %19 = loom.semaphore_take %18 : memref<512x512xf16> -> memref<512x512xf16>
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %23 = arith.muli %arg5, %c256 : index
          %24 = arith.muli %arg3, %c262144 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [512, 256], strides: [512, 1] : memref<4096x512xf16> to memref<512x256xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<512x256xf16, strided<[512, 1], offset: ?>> to memref<512x256xf16>
          %26 = arith.muli %arg4, %c512 : index
          %27 = arith.muli %arg5, %c1048576 : index
          %28 = arith.addi %27, %26 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [256, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<256x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<256x512xf16, strided<[4096, 1], offset: ?>> to memref<256x512xf16>
          loom.matmul ins(%17, %15 : memref<512x256xf16>, memref<256x512xf16>) outs(%19 : memref<512x512xf16>)
          loom.semaphore_give %15 : memref<256x512xf16>
          loom.semaphore_give %17 : memref<512x256xf16>
        }
        %cast = memref.cast %19 : memref<512x512xf16> to memref<?x?xf16>
        %20 = arith.muli %arg4, %c512 : index
        %21 = arith.muli %arg3, %c2097152 : index
        %22 = arith.addi %21, %20 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f10__y_n_n__block_size_0512__block_size_1512__block_size_2256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
        %15 = loom.semaphore_take %14 : memref<256x512xf16> -> memref<256x512xf16>
        %16 = loom.alloc [512, 256] on @L1 : memref<512x256xf16>
        %17 = loom.semaphore_take %16 : memref<512x256xf16> -> memref<512x256xf16>
        %18 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %19 = loom.semaphore_take %18 : memref<512x512xf16> -> memref<512x512xf16>
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %23 = arith.muli %arg5, %c256 : index
          %24 = arith.muli %arg3, %c262144 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [512, 256], strides: [512, 1] : memref<4096x512xf16> to memref<512x256xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<512x256xf16, strided<[512, 1], offset: ?>> to memref<512x256xf16>
          %26 = arith.muli %arg4, %c512 : index
          %27 = arith.muli %arg5, %c1048576 : index
          %28 = arith.addi %27, %26 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [256, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<256x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<256x512xf16, strided<[4096, 1], offset: ?>> to memref<256x512xf16>
          loom.matmul ins(%17, %15 : memref<512x256xf16>, memref<256x512xf16>) outs(%19 : memref<512x512xf16>)
          loom.semaphore_give %15 : memref<256x512xf16>
          loom.semaphore_give %17 : memref<512x256xf16>
        }
        %cast = memref.cast %19 : memref<512x512xf16> to memref<?x?xf16>
        %20 = arith.muli %arg4, %c512 : index
        %21 = arith.muli %arg3, %c2097152 : index
        %22 = arith.addi %21, %20 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f10__y_x_n__block_size_0512__block_size_1512__block_size_2256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
        %15 = loom.semaphore_take %14 : memref<256x512xf16> -> memref<256x512xf16>
        %16 = loom.alloc [512, 256] on @L1 : memref<512x256xf16>
        %17 = loom.semaphore_take %16 : memref<512x256xf16> -> memref<512x256xf16>
        %18 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
        %19 = loom.semaphore_take %18 : memref<512x512xf16> -> memref<512x512xf16>
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %23 = arith.muli %arg5, %c256 : index
          %24 = arith.muli %arg3, %c262144 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [512, 256], strides: [512, 1] : memref<4096x512xf16> to memref<512x256xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<512x256xf16, strided<[512, 1], offset: ?>> to memref<512x256xf16>
          %26 = arith.muli %arg4, %c512 : index
          %27 = arith.muli %arg5, %c1048576 : index
          %28 = arith.addi %27, %26 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [256, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<256x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<256x512xf16, strided<[4096, 1], offset: ?>> to memref<256x512xf16>
          loom.matmul ins(%17, %15 : memref<512x256xf16>, memref<256x512xf16>) outs(%19 : memref<512x512xf16>)
          loom.semaphore_give %15 : memref<256x512xf16>
          loom.semaphore_give %17 : memref<512x256xf16>
        }
        %cast = memref.cast %19 : memref<512x512xf16> to memref<?x?xf16>
        %20 = arith.muli %arg4, %c512 : index
        %21 = arith.muli %arg3, %c2097152 : index
        %22 = arith.addi %21, %20 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [512, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<512x512xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f01__n_n_n__block_size_04096__block_size_164__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        %20 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %21 = loom.semaphore_take %20 : memref<4096x64xf16> -> memref<4096x64xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [4096, 64], strides: [512, 1] : memref<4096x512xf16> to memref<4096x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<4096x64xf16, strided<[512, 1], offset: ?>> to memref<4096x64xf16>
          %24 = arith.muli %15, %c64 : index
          %25 = arith.muli %arg5, %c262144 : index
          %26 = arith.addi %25, %24 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>> to memref<64x64xf16>
          loom.matmul ins(%19, %17 : memref<4096x64xf16>, memref<64x64xf16>) outs(%21 : memref<4096x64xf16>)
          loom.semaphore_give %17 : memref<64x64xf16>
          loom.semaphore_give %19 : memref<4096x64xf16>
        }
        %cast = memref.cast %21 : memref<4096x64xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c64 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f01__y_n_n__block_size_04096__block_size_164__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        %20 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %21 = loom.semaphore_take %20 : memref<4096x64xf16> -> memref<4096x64xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [4096, 64], strides: [512, 1] : memref<4096x512xf16> to memref<4096x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<4096x64xf16, strided<[512, 1], offset: ?>> to memref<4096x64xf16>
          %24 = arith.muli %15, %c64 : index
          %25 = arith.muli %arg5, %c262144 : index
          %26 = arith.addi %25, %24 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>> to memref<64x64xf16>
          loom.matmul ins(%19, %17 : memref<4096x64xf16>, memref<64x64xf16>) outs(%21 : memref<4096x64xf16>)
          loom.semaphore_give %17 : memref<64x64xf16>
          loom.semaphore_give %19 : memref<4096x64xf16>
        }
        %cast = memref.cast %21 : memref<4096x64xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c64 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f01__x_n_n__block_size_04096__block_size_164__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        %20 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %21 = loom.semaphore_take %20 : memref<4096x64xf16> -> memref<4096x64xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [4096, 64], strides: [512, 1] : memref<4096x512xf16> to memref<4096x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<4096x64xf16, strided<[512, 1], offset: ?>> to memref<4096x64xf16>
          %24 = arith.muli %15, %c64 : index
          %25 = arith.muli %arg5, %c262144 : index
          %26 = arith.addi %25, %24 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>> to memref<64x64xf16>
          loom.matmul ins(%19, %17 : memref<4096x64xf16>, memref<64x64xf16>) outs(%21 : memref<4096x64xf16>)
          loom.semaphore_give %17 : memref<64x64xf16>
          loom.semaphore_give %19 : memref<4096x64xf16>
        }
        %cast = memref.cast %21 : memref<4096x64xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c64 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f01__a_n_n__block_size_02048__block_size_164__block_size_2256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c8388608 = arith.constant 8388608 : index
      %c1048576 = arith.constant 1048576 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [256, 64] on @L1 : memref<256x64xf16>
          %17 = loom.semaphore_take %16 : memref<256x64xf16> -> memref<256x64xf16>
          %18 = loom.alloc [2048, 256] on @L1 : memref<2048x256xf16>
          %19 = loom.semaphore_take %18 : memref<2048x256xf16> -> memref<2048x256xf16>
          %20 = loom.alloc [2048, 64] on @L1 : memref<2048x64xf16>
          %21 = loom.semaphore_take %20 : memref<2048x64xf16> -> memref<2048x64xf16>
          scf.for %arg6 = %c0 to %c2 step %c1 {
            %25 = arith.muli %arg6, %c256 : index
            %26 = arith.muli %arg5, %c1048576 : index
            %27 = arith.addi %26, %25 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [2048, 256], strides: [512, 1] : memref<4096x512xf16> to memref<2048x256xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<2048x256xf16, strided<[512, 1], offset: ?>> to memref<2048x256xf16>
            %28 = arith.muli %15, %c64 : index
            %29 = arith.muli %arg6, %c1048576 : index
            %30 = arith.addi %29, %28 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [256, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<256x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<256x64xf16, strided<[4096, 1], offset: ?>> to memref<256x64xf16>
            loom.matmul ins(%19, %17 : memref<2048x256xf16>, memref<256x64xf16>) outs(%21 : memref<2048x64xf16>)
            loom.semaphore_give %17 : memref<256x64xf16>
            loom.semaphore_give %19 : memref<2048x256xf16>
          }
          %cast = memref.cast %21 : memref<2048x64xf16> to memref<?x?xf16>
          %22 = arith.muli %15, %c64 : index
          %23 = arith.muli %arg5, %c8388608 : index
          %24 = arith.addi %23, %22 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [2048, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<2048x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<2048x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<2048x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f10__n_n_n__block_size_04096__block_size_164__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        %20 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %21 = loom.semaphore_take %20 : memref<4096x64xf16> -> memref<4096x64xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [4096, 64], strides: [512, 1] : memref<4096x512xf16> to memref<4096x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<4096x64xf16, strided<[512, 1], offset: ?>> to memref<4096x64xf16>
          %24 = arith.muli %15, %c64 : index
          %25 = arith.muli %arg5, %c262144 : index
          %26 = arith.addi %25, %24 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>> to memref<64x64xf16>
          loom.matmul ins(%19, %17 : memref<4096x64xf16>, memref<64x64xf16>) outs(%21 : memref<4096x64xf16>)
          loom.semaphore_give %17 : memref<64x64xf16>
          loom.semaphore_give %19 : memref<4096x64xf16>
        }
        %cast = memref.cast %21 : memref<4096x64xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c64 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f10__y_n_n__block_size_04096__block_size_164__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        %20 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %21 = loom.semaphore_take %20 : memref<4096x64xf16> -> memref<4096x64xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [4096, 64], strides: [512, 1] : memref<4096x512xf16> to memref<4096x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<4096x64xf16, strided<[512, 1], offset: ?>> to memref<4096x64xf16>
          %24 = arith.muli %15, %c64 : index
          %25 = arith.muli %arg5, %c262144 : index
          %26 = arith.addi %25, %24 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>> to memref<64x64xf16>
          loom.matmul ins(%19, %17 : memref<4096x64xf16>, memref<64x64xf16>) outs(%21 : memref<4096x64xf16>)
          loom.semaphore_give %17 : memref<64x64xf16>
          loom.semaphore_give %19 : memref<4096x64xf16>
        }
        %cast = memref.cast %21 : memref<4096x64xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c64 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f10__x_n_n__block_size_04096__block_size_164__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        %20 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %21 = loom.semaphore_take %20 : memref<4096x64xf16> -> memref<4096x64xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [4096, 64], strides: [512, 1] : memref<4096x512xf16> to memref<4096x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<4096x64xf16, strided<[512, 1], offset: ?>> to memref<4096x64xf16>
          %24 = arith.muli %15, %c64 : index
          %25 = arith.muli %arg5, %c262144 : index
          %26 = arith.addi %25, %24 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>> to memref<64x64xf16>
          loom.matmul ins(%19, %17 : memref<4096x64xf16>, memref<64x64xf16>) outs(%21 : memref<4096x64xf16>)
          loom.semaphore_give %17 : memref<64x64xf16>
          loom.semaphore_give %19 : memref<4096x64xf16>
        }
        %cast = memref.cast %21 : memref<4096x64xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c64 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f10__a_n_n__block_size_01024__block_size_164__block_size_2512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c4194304 = arith.constant 4194304 : index
      %c524288 = arith.constant 524288 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
          %17 = loom.semaphore_take %16 : memref<512x64xf16> -> memref<512x64xf16>
          %18 = loom.alloc [1024, 512] on @L1 : memref<1024x512xf16>
          %19 = loom.semaphore_take %18 : memref<1024x512xf16> -> memref<1024x512xf16>
          %20 = loom.alloc [1024, 64] on @L1 : memref<1024x64xf16>
          %21 = loom.semaphore_take %20 : memref<1024x64xf16> -> memref<1024x64xf16>
          %22 = arith.muli %arg5, %c524288 : index
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [1024, 512], strides: [512, 1] : memref<4096x512xf16> to memref<1024x512xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<1024x512xf16, strided<[512, 1], offset: ?>> to memref<1024x512xf16>
          %23 = arith.muli %15, %c64 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%23], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
          loom.matmul ins(%19, %17 : memref<1024x512xf16>, memref<512x64xf16>) outs(%21 : memref<1024x64xf16>)
          loom.semaphore_give %17 : memref<512x64xf16>
          loom.semaphore_give %19 : memref<1024x512xf16>
          %cast = memref.cast %21 : memref<1024x64xf16> to memref<?x?xf16>
          %24 = arith.muli %arg5, %c4194304 : index
          %25 = arith.addi %24, %23 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1024, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1024x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1024x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<1024x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f01__n_n_n__block_size_04096__block_size_164__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        %20 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %21 = loom.semaphore_take %20 : memref<4096x64xf16> -> memref<4096x64xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [4096, 64], strides: [512, 1] : memref<4096x512xf16> to memref<4096x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<4096x64xf16, strided<[512, 1], offset: ?>> to memref<4096x64xf16>
          %24 = arith.muli %15, %c64 : index
          %25 = arith.muli %arg5, %c262144 : index
          %26 = arith.addi %25, %24 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>> to memref<64x64xf16>
          loom.matmul ins(%19, %17 : memref<4096x64xf16>, memref<64x64xf16>) outs(%21 : memref<4096x64xf16>)
          loom.semaphore_give %17 : memref<64x64xf16>
          loom.semaphore_give %19 : memref<4096x64xf16>
        }
        %cast = memref.cast %21 : memref<4096x64xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c64 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f01__y_n_n__block_size_04096__block_size_164__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        %20 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %21 = loom.semaphore_take %20 : memref<4096x64xf16> -> memref<4096x64xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [4096, 64], strides: [512, 1] : memref<4096x512xf16> to memref<4096x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<4096x64xf16, strided<[512, 1], offset: ?>> to memref<4096x64xf16>
          %24 = arith.muli %15, %c64 : index
          %25 = arith.muli %arg5, %c262144 : index
          %26 = arith.addi %25, %24 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>> to memref<64x64xf16>
          loom.matmul ins(%19, %17 : memref<4096x64xf16>, memref<64x64xf16>) outs(%21 : memref<4096x64xf16>)
          loom.semaphore_give %17 : memref<64x64xf16>
          loom.semaphore_give %19 : memref<4096x64xf16>
        }
        %cast = memref.cast %21 : memref<4096x64xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c64 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f01__x_n_n__block_size_04096__block_size_164__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        %20 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %21 = loom.semaphore_take %20 : memref<4096x64xf16> -> memref<4096x64xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [4096, 64], strides: [512, 1] : memref<4096x512xf16> to memref<4096x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<4096x64xf16, strided<[512, 1], offset: ?>> to memref<4096x64xf16>
          %24 = arith.muli %15, %c64 : index
          %25 = arith.muli %arg5, %c262144 : index
          %26 = arith.addi %25, %24 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>> to memref<64x64xf16>
          loom.matmul ins(%19, %17 : memref<4096x64xf16>, memref<64x64xf16>) outs(%21 : memref<4096x64xf16>)
          loom.semaphore_give %17 : memref<64x64xf16>
          loom.semaphore_give %19 : memref<4096x64xf16>
        }
        %cast = memref.cast %21 : memref<4096x64xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c64 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f01__a_n_n__block_size_02048__block_size_164__block_size_2256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c8388608 = arith.constant 8388608 : index
      %c1048576 = arith.constant 1048576 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [256, 64] on @L1 : memref<256x64xf16>
          %17 = loom.semaphore_take %16 : memref<256x64xf16> -> memref<256x64xf16>
          %18 = loom.alloc [2048, 256] on @L1 : memref<2048x256xf16>
          %19 = loom.semaphore_take %18 : memref<2048x256xf16> -> memref<2048x256xf16>
          %20 = loom.alloc [2048, 64] on @L1 : memref<2048x64xf16>
          %21 = loom.semaphore_take %20 : memref<2048x64xf16> -> memref<2048x64xf16>
          scf.for %arg6 = %c0 to %c2 step %c1 {
            %25 = arith.muli %arg6, %c256 : index
            %26 = arith.muli %arg5, %c1048576 : index
            %27 = arith.addi %26, %25 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [2048, 256], strides: [512, 1] : memref<4096x512xf16> to memref<2048x256xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<2048x256xf16, strided<[512, 1], offset: ?>> to memref<2048x256xf16>
            %28 = arith.muli %15, %c64 : index
            %29 = arith.muli %arg6, %c1048576 : index
            %30 = arith.addi %29, %28 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [256, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<256x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<256x64xf16, strided<[4096, 1], offset: ?>> to memref<256x64xf16>
            loom.matmul ins(%19, %17 : memref<2048x256xf16>, memref<256x64xf16>) outs(%21 : memref<2048x64xf16>)
            loom.semaphore_give %17 : memref<256x64xf16>
            loom.semaphore_give %19 : memref<2048x256xf16>
          }
          %cast = memref.cast %21 : memref<2048x64xf16> to memref<?x?xf16>
          %22 = arith.muli %15, %c64 : index
          %23 = arith.muli %arg5, %c8388608 : index
          %24 = arith.addi %23, %22 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [2048, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<2048x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<2048x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<2048x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f10__n_n_n__block_size_04096__block_size_164__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        %20 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %21 = loom.semaphore_take %20 : memref<4096x64xf16> -> memref<4096x64xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [4096, 64], strides: [512, 1] : memref<4096x512xf16> to memref<4096x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<4096x64xf16, strided<[512, 1], offset: ?>> to memref<4096x64xf16>
          %24 = arith.muli %15, %c64 : index
          %25 = arith.muli %arg5, %c262144 : index
          %26 = arith.addi %25, %24 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>> to memref<64x64xf16>
          loom.matmul ins(%19, %17 : memref<4096x64xf16>, memref<64x64xf16>) outs(%21 : memref<4096x64xf16>)
          loom.semaphore_give %17 : memref<64x64xf16>
          loom.semaphore_give %19 : memref<4096x64xf16>
        }
        %cast = memref.cast %21 : memref<4096x64xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c64 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f10__y_n_n__block_size_04096__block_size_164__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        %20 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %21 = loom.semaphore_take %20 : memref<4096x64xf16> -> memref<4096x64xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [4096, 64], strides: [512, 1] : memref<4096x512xf16> to memref<4096x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<4096x64xf16, strided<[512, 1], offset: ?>> to memref<4096x64xf16>
          %24 = arith.muli %15, %c64 : index
          %25 = arith.muli %arg5, %c262144 : index
          %26 = arith.addi %25, %24 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>> to memref<64x64xf16>
          loom.matmul ins(%19, %17 : memref<4096x64xf16>, memref<64x64xf16>) outs(%21 : memref<4096x64xf16>)
          loom.semaphore_give %17 : memref<64x64xf16>
          loom.semaphore_give %19 : memref<4096x64xf16>
        }
        %cast = memref.cast %21 : memref<4096x64xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c64 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f10__x_n_n__block_size_04096__block_size_164__block_size_264(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %14 = arith.muli %arg3, %c8 overflow<nsw> : index
        %15 = arith.addi %14, %arg4 : index
        %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        %20 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %21 = loom.semaphore_take %20 : memref<4096x64xf16> -> memref<4096x64xf16>
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %23 = arith.muli %arg5, %c64 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [4096, 64], strides: [512, 1] : memref<4096x512xf16> to memref<4096x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<4096x64xf16, strided<[512, 1], offset: ?>> to memref<4096x64xf16>
          %24 = arith.muli %15, %c64 : index
          %25 = arith.muli %arg5, %c262144 : index
          %26 = arith.addi %25, %24 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>> to memref<64x64xf16>
          loom.matmul ins(%19, %17 : memref<4096x64xf16>, memref<64x64xf16>) outs(%21 : memref<4096x64xf16>)
          loom.semaphore_give %17 : memref<64x64xf16>
          loom.semaphore_give %19 : memref<4096x64xf16>
        }
        %cast = memref.cast %21 : memref<4096x64xf16> to memref<?x?xf16>
        %22 = arith.muli %15, %c64 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %21 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f10__a_n_n__block_size_02048__block_size_164__block_size_2256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c8388608 = arith.constant 8388608 : index
      %c1048576 = arith.constant 1048576 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [256, 64] on @L1 : memref<256x64xf16>
          %17 = loom.semaphore_take %16 : memref<256x64xf16> -> memref<256x64xf16>
          %18 = loom.alloc [2048, 256] on @L1 : memref<2048x256xf16>
          %19 = loom.semaphore_take %18 : memref<2048x256xf16> -> memref<2048x256xf16>
          %20 = loom.alloc [2048, 64] on @L1 : memref<2048x64xf16>
          %21 = loom.semaphore_take %20 : memref<2048x64xf16> -> memref<2048x64xf16>
          scf.for %arg6 = %c0 to %c2 step %c1 {
            %25 = arith.muli %arg6, %c256 : index
            %26 = arith.muli %arg5, %c1048576 : index
            %27 = arith.addi %26, %25 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [2048, 256], strides: [512, 1] : memref<4096x512xf16> to memref<2048x256xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<2048x256xf16, strided<[512, 1], offset: ?>> to memref<2048x256xf16>
            %28 = arith.muli %15, %c64 : index
            %29 = arith.muli %arg6, %c1048576 : index
            %30 = arith.addi %29, %28 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [256, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<256x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<256x64xf16, strided<[4096, 1], offset: ?>> to memref<256x64xf16>
            loom.matmul ins(%19, %17 : memref<2048x256xf16>, memref<256x64xf16>) outs(%21 : memref<2048x64xf16>)
            loom.semaphore_give %17 : memref<256x64xf16>
            loom.semaphore_give %19 : memref<2048x256xf16>
          }
          %cast = memref.cast %21 : memref<2048x64xf16> to memref<?x?xf16>
          %22 = arith.muli %15, %c64 : index
          %23 = arith.muli %arg5, %c8388608 : index
          %24 = arith.addi %23, %22 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [2048, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<2048x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<2048x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<2048x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
}
