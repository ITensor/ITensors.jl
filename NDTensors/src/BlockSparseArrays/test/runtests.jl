using Test
using NDTensors.BlockSparseArrays
using BlockArrays: BlockArrays

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
end
