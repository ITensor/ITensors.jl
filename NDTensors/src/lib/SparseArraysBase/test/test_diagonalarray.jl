@eval module $(gensym())
using LinearAlgebra: norm
using NDTensors.SparseArraysBase: SparseArraysBase
include("SparseArraysBaseTestUtils/SparseArraysBaseTestUtils.jl")
using .SparseArraysBaseTestUtils.DiagonalArrays: DiagonalArray
using Test: @test, @testset, @test_throws
@testset "DiagonalArray (eltype=$elt)" for elt in (Float32, ComplexF32, Float64, ComplexF64)
  # TODO: Test `fill!`.

  # Test
  a = DiagonalArray{elt}(undef, 2, 3)
  @test size(a) == (2, 3)
  a[1, 1] = 11
  a[2, 2] = 22
  @test a[1, 1] == 11
  @test a[2, 2] == 22
  @test_throws ArgumentError a[1, 2] = 12
  @test SparseArraysBase.storage_indices(a) == 1:2
  @test collect(SparseArraysBase.stored_indices(a)) ==
    [CartesianIndex(1, 1), CartesianIndex(2, 2)]
  a[1, 2] = 0
  @test a[1, 1] == 11
  @test a[2, 2] == 22

  a_dense = SparseArraysBase.densearray(a)
  @test a_dense == a
  @test a_dense isa Array{elt,ndims(a)}

  b = similar(a)
  @test b isa DiagonalArray
  @test size(b) == (2, 3)

  a = DiagonalArray(elt[1, 2, 3], (3, 3))
  @test size(a) == (3, 3)
  @test a[1, 1] == 1
  @test a[2, 2] == 2
  @test a[3, 3] == 3
  @test a[SparseArraysBase.StorageIndex(1)] == 1
  @test a[SparseArraysBase.StorageIndex(2)] == 2
  @test a[SparseArraysBase.StorageIndex(3)] == 3
  @test iszero(a[1, 2])

  a = DiagonalArray(elt[1, 2, 3], (3, 3))
  a = 2 * a
  @test size(a) == (3, 3)
  @test a[1, 1] == 2
  @test a[2, 2] == 4
  @test a[3, 3] == 6
  @test iszero(a[1, 2])

  a = DiagonalArray(elt[1, 2, 3], (3, 3))
  a_r = reshape(a, 9)
  @test a_r isa DiagonalArray{elt,1}
  for I in LinearIndices(a)
    @test a[I] == a_r[I]
  end

  # This needs `Base.reshape` with a custom destination
  # calling `SparseArraysBase.sparse_reshape!`
  # in order to specify an appropriate output
  # type to work.
  a = DiagonalArray(elt[1, 2], (2, 2, 2))
  a_r = reshape(a, 2, 4)
  @test a_r isa Matrix{elt}
  for I in LinearIndices(a)
    @test a[I] == a_r[I]
  end

  # Matrix multiplication!
  a1 = DiagonalArray(elt[1, 2], (2, 2))
  a2 = DiagonalArray(elt[2, 3], (2, 2))
  a_dest = a1 * a2
  @test Array(a_dest) ≈ Array(a1) * Array(a2)
  @test a_dest isa DiagonalArray{elt}
end
end
