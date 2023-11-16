using Test: @test, @testset
using BlockArrays: BlockArrays, BlockRange, blocksize
using LinearAlgebra: Diagonal, Hermitian, eigen, qr
using NDTensors: contract
using NDTensors.BlockSparseArrays:
  BlockSparseArrays, BlockSparseArray, gradedrange, nonzero_blockkeys, fusedims
using ITensors: QN

include("TestBlockSparseArraysUtils.jl")

@testset "Test NDTensors.BlockSparseArrays" begin
  @testset "README" begin
    @test include(
      joinpath(
        pkgdir(BlockSparseArrays), "src", "BlockSparseArrays", "examples", "README.jl"
      ),
    ) isa Any
  end
  @testset "exports $s" for s in [:BlockSparseArray]
    @test Base.isexported(BlockSparseArrays, s)
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
  @testset "contract (eltype=$elt)" for elt in (Float32, ComplexF32, Float64, ComplexF64)
    d1, d2, d3, d4 = [2, 3], [3, 4], [4, 5], [5, 6]
    a1 = BlockSparseArray{elt}(d1, d2, d3)
    TestBlockSparseArraysUtils.set_blocks!(a1, randn, b -> iseven(sum(b.n)))
    a2 = BlockSparseArray{elt}(d2, d4)
    TestBlockSparseArraysUtils.set_blocks!(a2, randn, b -> iseven(sum(b.n)))
    a_dest, labels_dest = contract(a1, (1, -1, 2), a2, (-1, 3))
    @test labels_dest == (1, 2, 3)
    @test eltype(a_dest) == elt
    # TODO: Output `labels_dest` as well.
    a_dest_dense = contract(Array(a1), (1, -1, 2), Array(a2), (-1, 3))
    @test a_dest ≈ a_dest_dense
  end
  @testset "qr (eltype=$elt, dims=$d)" for elt in
                                           (Float32, ComplexF32, Float64, ComplexF64),
    d in (([3, 4], [2, 3]), ([2, 3], [3, 4]), ([2, 3], [3]), ([3, 4], [2]), ([2], [3, 4]))

    a = BlockSparseArray{elt}(d)
    TestBlockSparseArraysUtils.set_blocks!(a, randn, b -> iseven(sum(b.n)))
    q, r = qr(a)
    @test q * r ≈ a
  end
  @testset "eigen (eltype=$elt)" for elt in (Float32, ComplexF32, Float64, ComplexF64)
    d1, d2 = [2, 3], [2, 3]
    a = BlockSparseArray{elt}(d1, d2)
    TestBlockSparseArraysUtils.set_blocks!(a, randn, b -> allequal(b.n))
    d, u = eigen(Hermitian(a))
    @test eltype(d) == real(elt)
    @test eltype(u) == elt
    @test Hermitian(Matrix(a)) * Matrix(u) ≈ Matrix(u) * Diagonal(Vector(d))
  end
end
