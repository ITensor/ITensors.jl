using Test
using BlockArrays: BlockArrays, blockedrange, blocksize
using NDTensors.BlockSparseArrays:
  BlockSparseArrays, BlockSparseArray, gradedrange, nonzero_blockkeys, fusedims
using ITensors: QN

@testset "Test NDTensors.BlockSparseArrays" begin
  @testset "README" begin
    @test include(
      joinpath(
        pkgdir(BlockSparseArrays), "src", "BlockSparseArrays", "examples", "README.jl"
      ),
    ) isa Any
  end
  @testset "Mixed block test" begin
    blocks = [BlockArrays.Block(1, 1), BlockArrays.Block(2, 2)]
    block_data = [randn(2, 2)', randn(3, 3)]
    inds = ([2, 3], [2, 3])
    A = BlockSparseArray(blocks, block_data, inds)
    @test A[BlockArrays.Block(1, 1)] == block_data[1]
    @test A[BlockArrays.Block(1, 2)] == zeros(2, 3)
  end
  @testset "fusedims" begin
    elt = Float64
    d = [2, 3]
    sectors = [QN(0, 2), QN(1, 2)]
    i = gradedrange(d, sectors)

    B = BlockSparseArray{elt}(i, i, i, i)
    B[BlockArrays.Block(1, 1, 1, 1)] = randn(2, 2, 2, 2)
    B[BlockArrays.Block(2, 2, 2, 2)] = randn(3, 3, 3, 3)
    @test size(B) == (5, 5, 5, 5)
    @test blocksize(B) == (2, 2, 2, 2)
    # TODO: Define `nnz`.
    @test length(nonzero_blockkeys(B)) == 2

    B_sub = B[
      [BlockArrays.Block(2)],
      [BlockArrays.Block(2)],
      [BlockArrays.Block(2)],
      [BlockArrays.Block(2)],
    ]
    @test B_sub isa BlockSparseArray{elt,4}
    @test B[BlockArrays.Block(2, 2, 2, 2)] == B_sub[BlockArrays.Block(1, 1, 1, 1)]
    @test size(B_sub) == (3, 3, 3, 3)
    @test blocksize(B_sub) == (1, 1, 1, 1)
    # TODO: Define `nnz`.
    @test length(nonzero_blockkeys(B_sub)) == 1

    B_fused = fusedims(B, (1, 2), (3, 4))
    @test B_fused isa BlockSparseArray{elt,2}
    @test size(B_fused) == (25, 25)
    @test blocksize(B_fused) == (2, 2)
    # TODO: Define `nnz`.
    # This is broken because it allocates all blocks,
    # need to fix that.
    @test_broken length(nonzero_blockkeys(B_fused)) == 2
  end
end
