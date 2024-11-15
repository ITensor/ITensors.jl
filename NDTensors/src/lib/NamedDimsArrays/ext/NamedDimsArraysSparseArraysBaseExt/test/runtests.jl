@eval module $(gensym())
using LinearAlgebra: Diagonal
using Test: @test, @testset
using NDTensors.SparseArraysBase: densearray
using NDTensors.NamedDimsArrays: named, unname
@testset "NamedDimsArraysSparseArraysBaseExt (eltype=$elt)" for elt in (Float32, Float64)
  na = named(Diagonal(randn(2)), ("i", "j"))
  na_dense = densearray(na)
  @test na ≈ na_dense
  @test unname(na_dense) isa Array
end
end
