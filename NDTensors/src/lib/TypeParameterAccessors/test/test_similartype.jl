@eval module $(gensym())
using Test: @test, @test_broken, @testset
using LinearAlgebra: Adjoint
using NDTensors.TypeParameterAccessors: similartype
@testset "TypeParameterAccessors similartype" begin
  @test similartype(Array, Float64, (2, 2)) == Matrix{Float64}
  # TODO: Is this a good definition? Probably it should be left unspecified.
  @test similartype(Array) == Array{Any}
  @test similartype(Array, Float64) == Array{Float64}
  @test similartype(Array, (2, 2)) == Matrix
  @test similartype(Adjoint{Float32,Matrix{Float32}}, Float64, (2, 2, 2)) ==
    Array{Float64,3}
  @test similartype(Adjoint{Float32,Matrix{Float32}}, Float64) == Matrix{Float64}
end
end
