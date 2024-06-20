@eval module $(gensym())
using Test: @test_throws, @testset
using NDTensors.TypeParameterAccessors:
  TypeParameterAccessors,
  Position,
  TypeParameter,
  set_type_parameter,
  set_type_parameters,
  specify_type_parameter,
  specify_type_parameters,
  type_parameter,
  type_parameters,
  unspecify_type_parameter,
  unspecify_type_parameters
include("utils/test_inferred.jl")
@testset "TypeParameterAccessors basics" begin
  @testset "Get parameters" begin
    @test_inferred type_parameter(AbstractArray{Float64}, 1) == Float64 wrapped = true
    @test_inferred type_parameter(AbstractArray{Float64}, Position(1)) == Float64
    @test_inferred type_parameter(AbstractArray{Float64}, eltype) == Float64
    @test_inferred type_parameter(AbstractMatrix{Float64}, ndims) == 2

    @test_inferred type_parameter(Array{Float64}, 1) == Float64 wrapped = true
    @test_inferred type_parameter(Array{Float64}, Position(1)) == Float64
    @test_inferred type_parameter(Val{3}) == 3
    @test_throws ErrorException type_parameter(Array, 1)
    @test_inferred type_parameter(Array{Float64}, eltype) == Float64
    @test_inferred type_parameter(Matrix{Float64}, ndims) == 2
    @test_throws ErrorException type_parameter(Array{Float64}, ndims) == 2
    @test_inferred type_parameters(Matrix{Float64}, (2, eltype)) == (2, Float64) wrapped =
      true
    @test_inferred type_parameters(Matrix{Float64}, (Position(2), eltype)) == (2, Float64)
  end
  @testset "Set parameters" begin
    @test_inferred set_type_parameter(Array, 1, Float64) == Array{Float64} wrapped = true
    @test_inferred set_type_parameter(Array, Position(1), Float64) == Array{Float64}
    @test_inferred set_type_parameter(Array, 2, 2) == Matrix wrapped = true
    @test_inferred set_type_parameter(Array, eltype, Float32) == Array{Float32}
    @test_inferred set_type_parameters(
      Array, (eltype, Position(2)), (TypeParameter(Float32), TypeParameter(3))
    ) == Array{Float32,3}
    @test_inferred set_type_parameters(Array, (eltype, 2), (Float32, 3)) == Array{Float32,3} wrapped =
      true

    # TODO: This should infer without wrapping but doesn't.
    @test_inferred set_type_parameters(
      Array, (eltype, Position(2)), (Float32, TypeParameter(3))
    ) == Array{Float32,3} wrapped = true
  end
  @testset "Specify parameters" begin
    @test_inferred specify_type_parameter(Array, 1, Float64) == Array{Float64} wrapped =
      true
    @test_inferred specify_type_parameter(Array, Position(1), Float64) == Array{Float64}
    @test_inferred specify_type_parameters(Matrix, (2, 1), (4, Float32)) == Matrix{Float32} wrapped =
      true
    @test_inferred specify_type_parameters(Array, (Float64, 2)) == Matrix{Float64} wrapped =
      true
    @test_inferred specify_type_parameter(Array, eltype, Float32) == Array{Float32}
    @test_inferred specify_type_parameters(Array, (eltype, 2), (Float32, 3)) ==
      Array{Float32,3} wrapped = true
  end
  @testset "Unspecify parameters" begin
    @test_inferred unspecify_type_parameter(Vector, 2) == Array wrapped = true
    @test_inferred unspecify_type_parameter(Vector, Position(2)) == Array
    @test_inferred unspecify_type_parameter(Vector{Float64}, eltype) == Vector
    @test_inferred unspecify_type_parameters(Vector{Float64}) == Array
    @test_inferred unspecify_type_parameters(Vector{Float64}, (eltype, 2)) == Array wrapped =
      true
    @test_inferred unspecify_type_parameters(Vector{Float64}, (eltype, Position(2))) ==
      Array
  end
  @testset "On objects" begin
    @test_inferred type_parameter(Val{3}()) == 3
    @test_inferred type_parameter(Val{Float32}()) == Float32
    a = randn(Float32, (2, 2, 2))
    @test_inferred type_parameter(a, 1) == Float32 wrapped = true
    @test_inferred type_parameter(a, eltype) == Float32
    @test_inferred type_parameter(a, Position(1)) == Float32
    @test_inferred type_parameter(a, 2) == 3 wrapped = true
    @test_inferred type_parameter(a, ndims) == 3
    @test_inferred type_parameters(a) == (Float32, 3)
    @test_inferred type_parameters(a, (2, eltype)) == (3, Float32) wrapped = true
  end
end
end
