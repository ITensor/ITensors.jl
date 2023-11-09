using Test
using BlockArrays: BlockArrays, blockedrange, Block
using NDTensors.BlockSparseArrays: BlockSparseArrays, BlockSparseArray, gradedblockedrange, nonzero_blockkeys, fusedims, getindices
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
    blocks = [Block(1, 1), Block(2, 2)]
    block_data = [randn(2, 2)', randn(3, 3)]
    inds = ([2, 3], [2, 3])
    A = BlockSparseArray(blocks, block_data, inds)
    @test A[Block(1, 1)] == block_data[1]
    @test A[Block(1, 2)] == zeros(2, 3)
  end
  @testset "fusedims" begin
    d = [2, 3]
    i = blockedrange(d)
    sectors = [QN(0), QN(1)]
    ig = gradedblockedrange(d, sectors)

    B = BlockSparseArray{Float64}(ig, ig, ig, ig)
    B[Block(1, 1, 1, 1)] = randn(2, 2, 2, 2)
    B[Block(2, 2, 2, 2)] = randn(3, 3, 3, 3)
    @show axes(B)
    @show length(nonzero_blockkeys(B))

    B_sub = getindices(B, [Block(2)], [Block(2)], [Block(2)], [Block(2)])
    @show B[Block(2, 2, 2, 2)] == B_sub[Block(1, 1, 1, 1)]
    @show length(nonzero_blockkeys(B_sub))

    B_fused = fusedims(B, (1, 2), (3, 4))
    @show axes(B_fused)
    @show length(nonzero_blockkeys(B_fused))
  end
end
