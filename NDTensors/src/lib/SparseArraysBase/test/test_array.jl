@eval module $(gensym())
using NDTensors.SparseArraysBase: SparseArraysBase
include("SparseArraysBaseTestUtils/SparseArraysBaseTestUtils.jl")
using Test: @test, @testset
@testset "Array (eltype=$elt)" for elt in (Float32, ComplexF32, Float64, ComplexF64)
  a = randn(2, 3)
  @test SparseArraysBase.sparse_storage(a) == a
  @test SparseArraysBase.index_to_storage_index(a, CartesianIndex(1, 2)) ==
    CartesianIndex(1, 2)
  @test SparseArraysBase.storage_index_to_index(a, CartesianIndex(1, 2)) ==
    CartesianIndex(1, 2)
end
end
