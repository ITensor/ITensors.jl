@eval module $(gensym())
using BlockArrays: Block, BlockedUnitRange, blockedrange, blocklength, blocksize
using LinearAlgebra: mul!
using NDTensors.BlockSparseArrays: BlockSparseArray, block_nstored
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
  @testset "LinearAlgebra" begin
    a1 = BlockSparseArray{elt}([2, 3], [2, 3])
    a1[Block(1, 1)] = randn(elt, size(@view(a1[Block(1, 1)])))
    a2 = BlockSparseArray{elt}([2, 3], [2, 3])
    a2[Block(1, 1)] = randn(elt, size(@view(a1[Block(1, 1)])))
    @show typeof(a1 * a2)
  end
end
end
