@eval module $(gensym())
using BlockArrays: Block, BlockedUnitRange, blockedrange, blocklength, blocksize
using LinearAlgebra: mul!
using NDTensors.BlockSparseArrays: BlockSparseArray, block_nstored, block_reshape
using NDTensors.SparseArrayInterface: nstored
using Test: @test, @testset
include("TestBlockSparseArraysUtils.jl")
@testset "BlockSparseArrays (eltype=$elt)" for elt in
                                               (Float32, Float64, ComplexF32, ComplexF64)
  @testset "Basics" begin
    a = BlockSparseArray{elt}([2, 3], [2, 3])
    @test eltype(a) === elt
    @test axes(a) == (1:5, 1:5)
    @test all(aᵢ -> aᵢ isa BlockedUnitRange, axes(a))
    @test blocklength.(axes(a)) == (2, 2)
    @test blocksize(a) == (2, 2)
    @test size(a) == (5, 5)
    @test block_nstored(a) == 0
    @test iszero(a)
    @test all(I -> iszero(a[I]), eachindex(a))

    a = BlockSparseArray{elt}([2, 3], [2, 3])
    a[3, 3] = 33
    @test eltype(a) === elt
    @test axes(a) == (1:5, 1:5)
    @test all(aᵢ -> aᵢ isa BlockedUnitRange, axes(a))
    @test blocklength.(axes(a)) == (2, 2)
    @test blocksize(a) == (2, 2)
    @test size(a) == (5, 5)
    @test block_nstored(a) == 1
    @test !iszero(a)
    @test a[3, 3] == 33
    @test all(eachindex(a)) do I
      if I == CartesianIndex(3, 3)
        a[I] == 33
      else
        iszero(a[I])
      end
    end
  end
  @testset "Tensor algebra" begin
    a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
    a[Block(1, 2)] = randn(elt, size(@view(a[Block(1, 2)])))
    a[Block(2, 1)] = randn(elt, size(@view(a[Block(2, 1)])))
    @test eltype(a) == elt
    @test block_nstored(a) == 2
    @test nstored(a) == 2 * 4 + 3 * 3

    b = 2 * a
    @test Array(b) ≈ 2 * Array(a)
    @test eltype(b) == elt
    @test block_nstored(b) == 2
    @test nstored(b) == 2 * 4 + 3 * 3

    b = a + a
    @test Array(b) ≈ 2 * Array(a)
    @test eltype(b) == elt
    @test block_nstored(b) == 2
    @test nstored(b) == 2 * 4 + 3 * 3

    x = BlockSparseArray{elt}(undef, ([3, 4], [2, 3]))
    x[Block(1, 2)] = randn(elt, size(@view(x[Block(1, 2)])))
    x[Block(2, 1)] = randn(elt, size(@view(x[Block(2, 1)])))
    b = a .+ a .+ 3 .* PermutedDimsArray(x, (2, 1))
    @test Array(b) ≈ 2 * Array(a) + 3 * permutedims(Array(x), (2, 1))
    @test eltype(b) == elt
    @test block_nstored(b) == 2
    @test nstored(b) == 2 * 4 + 3 * 3

    b = permutedims(a, (2, 1))
    @test Array(b) ≈ permutedims(Array(a), (2, 1))
    @test eltype(b) == elt
    @test block_nstored(b) == 2
    @test nstored(b) == 2 * 4 + 3 * 3

    b = map(x -> 2x, a)
    @test Array(b) ≈ 2 * Array(a)
    @test eltype(b) == elt
    @test block_nstored(b) == 2
    @test nstored(b) == 2 * 4 + 3 * 3
  end
  @testset "LinearAlgebra" begin
    a1 = BlockSparseArray{elt}([2, 3], [2, 3])
    a1[Block(1, 1)] = randn(elt, size(@view(a1[Block(1, 1)])))
    a2 = BlockSparseArray{elt}([2, 3], [2, 3])
    a2[Block(1, 1)] = randn(elt, size(@view(a1[Block(1, 1)])))
    a_dest = a1 * a2
    @test Array(a_dest) ≈ Array(a1) * Array(a2)
    @test a_dest isa BlockSparseArray{elt}
    @test block_nstored(a_dest) == 1
  end
  @testset "block_reshape" begin
    a = BlockSparseArray{elt}(undef, ([3, 4], [2, 3]))
    a[Block(1, 2)] = randn(elt, size(@view(a[Block(1, 2)])))
    a[Block(2, 1)] = randn(elt, size(@view(a[Block(2, 1)])))
    b = block_reshape(a, [6, 8, 9, 12])
    @test reshape(a[Block(1, 2)], 9) == b[Block(3)]
    @test reshape(a[Block(2, 1)], 8) == b[Block(2)]
    @test block_nstored(b) == 2
    @test nstored(b) == 17
  end
end
end
