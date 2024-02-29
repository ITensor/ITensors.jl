@eval module $(gensym())
using Test: @test_throws, @testset
using NDTensors.TypeParameterAccessors:
  TypeParameterAccessors,
  Position,
  default_type_parameter,
  default_type_parameters,
  set_default_type_parameter,
  set_default_type_parameters,
  specify_default_type_parameter,
  specify_default_type_parameters
include("utils/test_inferred.jl")
@testset "TypeParameterAccessors defaults" begin
  @testset "Erroneously requires wrapping to infer" begin end
  @testset "Get defaults" begin
    @test_inferred default_type_parameter(Array, 1) == Float64 wrapped = true
    @test_inferred default_type_parameter(Array, Position(1)) == Float64
    @test_inferred default_type_parameter(Array, 2) == 1 wrapped = true
    @test_inferred default_type_parameter(Array, Position(2)) == 1
    @test_inferred default_type_parameters(Array) == (Float64, 1)
    @test_inferred default_type_parameters(Array, (2, 1)) == (1, Float64) wrapped = true
    @test_inferred default_type_parameters(Array, (Position(2), Position(1))) ==
      (1, Float64)
    @test_inferred default_type_parameters(Array, (ndims, eltype)) == (1, Float64)
  end
  @testset "Set defaults" begin
    @test_inferred set_default_type_parameter(Array{Float32}, 1) == Array{Float64} wrapped =
      true
    @test_inferred set_default_type_parameter(Array{Float32}, Position(1)) == Array{Float64}
    @test_inferred set_default_type_parameter(Array{Float32}, eltype) == Array{Float64}
    @test_inferred set_default_type_parameters(Array{Float32}) == Vector{Float64}
    @test_inferred set_default_type_parameters(Array{Float32}, (1, 2)) == Vector{Float64} wrapped =
      true
    @test_inferred set_default_type_parameters(
      Array{Float32}, (Position(1), Position(2))
    ) == Vector{Float64}
    @test_inferred set_default_type_parameters(Array{Float32}, (eltype, ndims)) ==
      Vector{Float64}
    @test_inferred set_default_type_parameters(Array) == Vector{Float64} wrapped = true
    @test_inferred set_default_type_parameters(Array, (Position(1),)) == Array{Float64}
    @test_inferred set_default_type_parameters(Array, (Position(1), Position(2))) ==
      Vector{Float64}
  end
  @testset "Specify defaults" begin
    @test_inferred specify_default_type_parameter(Array, 1) == Array{Float64} wrapped = true
    @test_inferred specify_default_type_parameter(Array, Position(1)) == Array{Float64}
    @test_inferred specify_default_type_parameter(Array, eltype) == Array{Float64}
    @test_inferred specify_default_type_parameter(Array, 2) == Vector wrapped = true
    @test_inferred specify_default_type_parameter(Array, Position(2)) == Vector
    @test_inferred specify_default_type_parameter(Array, ndims) == Vector
    @test_inferred specify_default_type_parameters(Array) == Vector{Float64}
    @test_inferred specify_default_type_parameters(Array, (1,)) == Array{Float64} wrapped =
      true
    @test_inferred specify_default_type_parameters(Array, (Position(1),)) == Array{Float64}
    @test_inferred specify_default_type_parameters(Array, (eltype,)) == Array{Float64}
    @test_inferred specify_default_type_parameters(Array, (2,)) == Vector wrapped = true
    @test_inferred specify_default_type_parameters(Array, (Position(2),)) == Vector
    @test_inferred specify_default_type_parameters(Array, (ndims,)) == Vector
    @test_inferred specify_default_type_parameters(Array, (1, 2)) == Vector{Float64} wrapped =
      true
    @test_inferred specify_default_type_parameters(Array, (Position(1), Position(2))) ==
      Vector{Float64}
    @test_inferred specify_default_type_parameters(Array, (eltype, ndims)) ==
      Vector{Float64}
  end
  @testset "On objects" begin
    a = randn(Float32, (2, 2, 2))
    @test_inferred default_type_parameter(a, 1) == Float64 wrapped = true
    @test_inferred default_type_parameter(a, eltype) == Float64
    @test_inferred default_type_parameter(a, 2) == 1 wrapped = true
    @test_inferred default_type_parameter(a, ndims) == 1
    @test_inferred default_type_parameters(a) == (Float64, 1)
  end
end
end
