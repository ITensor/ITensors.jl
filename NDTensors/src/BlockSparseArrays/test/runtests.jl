using Test
using BlockArrays: BlockArrays, BlockRange, blocksize
using NDTensors: contract
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
  @testset "contract" begin
    function randn_even_blocks!(a)
      for b in BlockRange(a)
        if iseven(sum(b.n))
          a[b] = randn(eltype(a), size(@view(a[b])))
        end
      end
    end

    d1, d2, d3, d4 = [2, 3], [3, 4], [4, 5], [5, 6]
    elt = Float64
    a1 = BlockSparseArray{elt}(d1, d2, d3)
    randn_even_blocks!(a1)
    a2 = BlockSparseArray{elt}(d2, d4)
    randn_even_blocks!(a2)
    a_dest, labels_dest = contract(a1, (1, -1, 2), a2, (-1, 3))
    @show labels_dest == (1, 2, 3)

    # TODO: Output `labels_dest` as well.
    a_dest_dense = contract(Array(a1), (1, -1, 2), Array(a2), (-1, 3))
    @show a_dest ≈ a_dest_dense
  end
end
