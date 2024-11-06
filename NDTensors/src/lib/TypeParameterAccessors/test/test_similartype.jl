@eval module $(gensym())
using Test: @test, @test_broken, @testset
using LinearAlgebra: Adjoint, Diagonal
using NDTensors.TypeParameterAccessors: NDims, similartype
@testset "TypeParameterAccessors similartype" begin
  @test similartype(Array, Float64, (2, 2)) == Matrix{Float64}
  @test similartype(Array) == Array
  @test similartype(Array, Float64) == Array{Float64}
  @test similartype(Array, (2, 2)) == Matrix
  @test similartype(Array, NDims(2)) == Matrix
  @test similartype(Array, Float64, (2, 2)) == Matrix{Float64}
  @test similartype(Array, Float64, NDims(2)) == Matrix{Float64}
  @test similartype(Adjoint{Float32,Matrix{Float32}}, Float64, (2, 2, 2)) ==
    Array{Float64,3}
  @test similartype(Adjoint{Float32,Matrix{Float32}}, Float64, NDims(3)) == Array{Float64,3}
  @test similartype(Adjoint{Float32,Matrix{Float32}}, Float64) == Matrix{Float64}
  @test similartype(Diagonal{Float32,Vector{Float32}}) == Matrix{Float32}
  @test similartype(Diagonal{Float32,Vector{Float32}}, Float64) == Matrix{Float64}
  @test similartype(Diagonal{Float32,Vector{Float32}}, (2, 2, 2)) == Array{Float32,3}
  @test similartype(Diagonal{Float32,Vector{Float32}}, NDims(3)) == Array{Float32,3}
  @test similartype(Diagonal{Float32,Vector{Float32}}, Float64, (2, 2, 2)) ==
    Array{Float64,3}
  @test similartype(Diagonal{Float32,Vector{Float32}}, Float64, NDims(3)) ==
    Array{Float64,3}
end
end
