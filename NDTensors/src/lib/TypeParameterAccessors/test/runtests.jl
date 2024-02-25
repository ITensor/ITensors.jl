@eval module $(gensym())
using Test: @test_throws, @testset
using NDTensors.TypeParameterAccessors:
  TypeParameterAccessors,
  Position,
  TypeParameter,
  UnspecifiedTypeParameter,
  default_type_parameter,
  default_type_parameters,
  set_default_parameter,
  set_default_parameters,
  set_parameter,
  set_parameters,
  specify_default_parameter,
  specify_default_parameters,
  specify_parameter,
  specify_parameters,
  type_parameter,
  type_parameters,
  unspecify_parameter,
  unspecify_parameters

include("test_inferred.jl")

@testset "TypeParameterAccessors" begin
  @test_inferred type_parameter(Array{Float64}, 1) == Float64 wrapped = true
  @test_inferred type_parameter(Array{Float64}, Position(1)) == Float64
  @test_inferred type_parameter(Val{3}) == 3
  @test_throws ErrorException type_parameter(Array, 1)
  @test_inferred set_parameter(Array, 1, Float64) == Array{Float64} wrapped = true
  @test_inferred set_parameter(Array, Position(1), Float64) == Array{Float64}
  @test_inferred set_parameter(Array, 2, 2) == Matrix wrapped = true
  @test_inferred specify_parameter(Array, 1, Float64) == Array{Float64} wrapped = true
  @test_inferred specify_parameter(Array, Position(1), Float64) == Array{Float64}
  @test_inferred specify_parameters(Matrix, (2, 1), (4, Float32)) == Matrix{Float32} wrapped =
    true
  @test_inferred specify_parameters(Array, (Float64, 2)) == Matrix{Float64} wrapped = true

  # Named positions
  @test_inferred type_parameter(Array{Float64}, eltype) == Float64
  @test_inferred type_parameter(Matrix{Float64}, ndims) == 2
  @test_throws ErrorException type_parameter(Array{Float64}, ndims) == 2
  @test_inferred type_parameters(Matrix{Float64}, (2, eltype)) == (2, Float64) wrapped =
    true
  @test_inferred type_parameters(Matrix{Float64}, (Position(2), eltype)) == (2, Float64)
  @test_inferred set_parameter(Array, eltype, Float32) == Array{Float32}
  @test_inferred specify_parameter(Array, eltype, Float32) == Array{Float32}
  @test_inferred set_parameters(
    Array, (eltype, Position(2)), (TypeParameter(Float32), TypeParameter(3))
  ) == Array{Float32,3}
  @test_inferred set_parameters(Array, (eltype, 2), (Float32, 3)) == Array{Float32,3} wrapped =
    true
  @test_inferred specify_parameters(Array, (eltype, 2), (Float32, 3)) == Array{Float32,3} wrapped =
    true

  # TODO: These should infer without wrapping but don't.
  @test_inferred set_parameters(
    Array, (eltype, Position(2)), (Float32, TypeParameter(3))
  ) == Array{Float32,3} wrapped = true

  # Default values
  @test_inferred default_type_parameter(Array, 1) == Float64 wrapped = true
  @test_inferred default_type_parameter(Array, Position(1)) == Float64
  @test_inferred default_type_parameter(Array, 2) == 1 wrapped = true
  @test_inferred default_type_parameter(Array, Position(2)) == 1
  @test_inferred specify_default_parameter(Array, 1) == Array{Float64} wrapped = true
  @test_inferred specify_default_parameter(Array, Position(1)) == Array{Float64}
  @test_inferred specify_default_parameter(Array, eltype) == Array{Float64}
  @test_inferred specify_default_parameter(Array, 2) == Vector wrapped = true

  # TODO: These should infer without wrapping but don't.
  @test_inferred specify_default_parameter(Array, Position(2)) == Vector wrapped = true
  @test_inferred specify_default_parameter(Array, ndims) == Vector wrapped = true

  @test_inferred specify_default_parameters(Array, (1,)) == Array{Float64} wrapped = true
  @test_inferred specify_default_parameters(Array, (2,)) == Vector wrapped = true
  @test_inferred set_default_parameter(Array{Float32}, 1) == Array{Float64} wrapped = true
  @test_inferred set_default_parameter(Array{Float32}, Position(1)) == Array{Float64}
  @test_inferred set_default_parameters(Array{Float32}, (1, 2)) == Vector{Float64} wrapped =
    true

  # TODO: These should infer without wrapping but don't.
  @test_inferred specify_default_parameters(Array) == Vector{Float64} wrapped = true
  @test_inferred specify_default_parameters(Array, (eltype,)) == Array{Float64} wrapped =
    true
  @test_inferred specify_default_parameters(Array, (Position(1),)) == Array{Float64} wrapped =
    true
  @test_inferred specify_default_parameters(Array, (Position(2),)) == Vector wrapped = true
  @test_inferred set_default_parameters(Array{Float32}) == Vector{Float64} wrapped = true

  @test_inferred unspecify_parameter(Vector, 2) == Array wrapped = true
  @test_inferred unspecify_parameter(Vector, Position(2)) == Array
  @test_inferred unspecify_parameter(Vector{Float64}, eltype) == Vector
  @test_inferred unspecify_parameters(Vector{Float64}) == Array
  @test_inferred unspecify_parameters(Vector{Float64}, (eltype, 2)) == Array wrapped = true
  @test_inferred unspecify_parameters(Vector{Float64}, (eltype, Position(2))) == Array

  # On objects
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
  @test_inferred default_type_parameter(a, 1) == Float64 wrapped = true
  @test_inferred default_type_parameter(a, eltype) == Float64
  @test_inferred default_type_parameter(a, 2) == 1 wrapped = true
  @test_inferred default_type_parameter(a, ndims) == 1
  @test_inferred default_type_parameters(a) == (Float64, 1)

  struct MyType{P1,P2} end
  TypeParameterAccessors.default_type_parameter(::Type{<:MyType}, ::Position{1}) = :P1

  @test_inferred default_type_parameter(MyType{<:Any,2}, 1) == :P1 wrapped = true
  @test_inferred default_type_parameter(MyType{<:Any,2}, Position(1)) == :P1
  @test_inferred default_type_parameter(MyType{<:Any,2}, 2) == UnspecifiedTypeParameter() wrapped =
    true
  @test_inferred default_type_parameter(MyType{<:Any,2}, Position(2)) ==
    UnspecifiedTypeParameter()
  @test_inferred set_default_parameter(MyType{<:Any,2}, Position(2)) == MyType
  @test_inferred specify_default_parameter(MyType{<:Any,2}, Position(2)) == MyType{<:Any,2}

  # TODO: These should infer without wrapping but don't.
  @test_inferred specify_default_parameters(MyType) == MyType{:P1} wrapped = true

  # TODO: Inference is broken for these, only testing correctness.
  @test_inferred default_type_parameters(MyType{<:Any,2}) ==
    (:P1, UnspecifiedTypeParameter()) inferred = false
  @test_inferred set_default_parameter(MyType{<:Any,2}, 2) == MyType inferred = false
  @test_inferred set_default_parameters(MyType{<:Any,2}) == MyType{:P1} inferred = false
  @test_inferred specify_default_parameter(MyType{<:Any,2}, 2) == MyType{<:Any,2} inferred =
    false
  @test_inferred specify_default_parameters(MyType{<:Any,2}) == MyType{:P1,2} inferred =
    false
end
end
