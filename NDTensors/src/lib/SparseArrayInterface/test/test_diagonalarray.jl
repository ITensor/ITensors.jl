@eval module $(gensym())
using LinearAlgebra: norm
using NDTensors.SparseArrayInterface: SparseArrayInterface
include("SparseArrayInterfaceTestUtils/SparseArrayInterfaceTestUtils.jl")
using .SparseArrayInterfaceTestUtils.DiagonalArrays: DiagonalArray
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
  @test SparseArrayInterface.storage_indices(a) == 1:2
  @test collect(SparseArrayInterface.stored_indices(a)) ==
    [CartesianIndex(1, 1), CartesianIndex(2, 2)]
  a[1, 2] = 0
  @test a[1, 1] == 11
  @test a[2, 2] == 22

  a_dense = SparseArrayInterface.densearray(a)
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
  @test a[SparseArrayInterface.StorageIndex(1)] == 1
  @test a[SparseArrayInterface.StorageIndex(2)] == 2
  @test a[SparseArrayInterface.StorageIndex(3)] == 3
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
  # calling `SparseArrayInterface.sparse_reshape!`
  # in order to specify an appropriate output
  # type to work.
  a = DiagonalArray(elt[1, 2], (2, 2, 2))
  a_r = reshape(a, 2, 4)
  @test a_r isa Matrix{elt}
  for I in LinearIndices(a)
    @test a[I] == a_r[I]
  end
end
end
