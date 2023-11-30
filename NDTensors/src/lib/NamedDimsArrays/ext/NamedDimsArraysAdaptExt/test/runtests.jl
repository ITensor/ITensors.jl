@eval module $(gensym())
using Test: @test, @testset
using Adapt: adapt
using NDTensors.NamedDimsArrays: named
@testset "NamedDimsArraysAdaptExt (eltype=$elt)" for elt in (Float32, Float64)
  na = named(randn(2, 2), ("i", "j"))
  na_complex = adapt(Array{complex(elt)}, na)
  @test na ≈ na_complex
  @test eltype(na_complex) === complex(elt)
end
end
