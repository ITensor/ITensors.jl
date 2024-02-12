@eval module $(gensym())
using Test: @inferred, @test, @test_broken, @testset
using NDTensors.TypeParameterAccessors
using LinearAlgebra: Transpose

@testset "Test NDTensors.TypeParameterAccessors" begin
  @testset "Get parameters" begin
    @test @inferred(parameters(Array{Float32,3})) == (Float32, 3)
    @test @inferred(parameters(Array{Float32,2}(undef, (2, 3)))) == (Float32, 2)
    @test @inferred((() -> parameter(Array{Float32,3}, 1))()) == Float32
    @test @inferred((() -> parameter(Array{Float32,3}, 2))()) == 3

    @test @inferred(parameter(Array{Float32,3}, Position(1))) == Float32
    @test @inferred(parameter(Array{Float32,3}, Position(2))) == 3
    @test nparameters(Array) == 2
    @test nparameters(Base.SubArray) == 5
  end

  @testset "Set parameter at position" begin
    @test @inferred((() -> set_parameters(Array{Float32,3}, (Float16, 2)))()) ==
      Array{Float16,2}
    @test @inferred((() -> set_parameters(Array, (Float32, 3)))()) == Array{Float32,3}
    @test @inferred(set_parameters(Array{Float32,2})) == Array{Float32,2}

    @test @inferred((() -> set_parameter(Array{Float32,3}, 1, Float16))()) ==
      Array{Float16,3}
    @test @inferred((() -> set_parameter(Array{Float32,3}, 2, 2))()) == Array{Float32,2}
    @test @inferred((() -> set_parameter(Array{Float32}, 1, Float16))()) == Array{Float16}
    @test @inferred((() -> set_parameter(Array{Float32}, 2, 2))()) == Array{Float32,2}
    @test @inferred((() -> set_parameter(Vector, 1, Float16))()) == Array{Float16,1}
    @test @inferred((() -> set_parameter(Vector, 2, 2))()) == Matrix
    @test @inferred((() -> set_parameter(Array, 1, Float16))()) == Array{Float16}
    @test @inferred((() -> set_parameter(Array, 2, 2))()) == Matrix

    @test @inferred((() -> set_parameter(Vector, Position(1), Float32))()) ==
      Vector{Float32}
    @test @inferred ((
      () -> set_parameter(Array{Float32}, Position(2), TypeParameter(2))
    )()) == Array{Float32,2}
    @test @inferred((() -> set_parameter(Array{Float32}, Float16))()) == Array{Float16}
    @test @inferred(
      (() -> set_parameter(Array{<:Any,2}, Position(2), TypeParameter(3)))()
    ) == Array{<:Any,3}
  end

  @testset "Set ndim and eltype" begin
    @test @inferred((() -> set_ndims(Array{Float32,2}, 3))()) == Array{Float32,3}
    @test @inferred((() -> set_eltype(Array{Float32,2}, Float16))()) == Array{Float16,2}
    @test @inferred((() -> set_ndims(Array, 2))()) == Matrix
    @test @inferred(((() -> set_eltype(Array, Float32))())) == Array{Float32}
    m = Transpose(Matrix{Float32}(undef, (2, 3)))
    @test (() -> set_eltype(typeof(m), Float16))() == Transpose{Float16,Matrix{Float16}}
    ## TODO This code does not infer the correct type but there aren't any allocations
    ## When it is called so I believe its actually working properly
    ## In a wrapped function on the command line this does however show the correct
    ## value when using code_warntype.
    @test ((() -> set_ndims(Array{<:Any,3}, 4))()) == Array{<:Any,4}
    @test ((() -> set_eltype(Array{<:Any,3}, Float16))()) == Array{Float16,3}
  end

  @testset "Set a parameter if it is unspecified" begin
    @test @inferred(specify_parameters(Array{Float32,3}, Float16)) == Array{Float32,3}
    @test @inferred((() -> specify_parameters(Array{Float32}, Float16))()) == Array{Float32}
    @test @inferred((() -> specify_parameters(Array, Float16))()) == Array{Float16}
    @test @inferred(specify_parameter(Array{Float32,3}, 1, Float16)) == Array{Float32,3}
    @test @inferred(specify_parameter(Array{Float32,3}, 2, 2)) == Array{Float32,3}
    @test @inferred((() -> specify_parameter(Array{Float32}, 1, Float16))()) ==
      Array{Float32}
    @test @inferred((() -> specify_parameter(Array{Float32}, 2, 2))()) == Array{Float32,2}
    @test @inferred((() -> specify_parameter(Array, 1, Float16))()) == Array{Float16}
    @test @inferred((() -> specify_parameter(Array, 2, 2))()) == Array{<:Any,2}

    ## TODO this is broken because of a type stability issue in Julia `Base`
    @test_broken @inferred((() -> specify_parameters(Array{<:Any,3}, Float16))()) ==
      Array{Float16,3}
    @test specify_parameter(Array{<:Any,3}, 1, Float16) == Array{Float16,3}
    @test specify_parameter(Array{<:Any,3}, 2, 2) == Array{<:Any,3}
    @test_broken @inferred(specify_parameter(Array{<:Any,3}, 1, Float16)) ==
      Array{Float16,3}
    @test_broken @inferred(specify_parameters(Array{<:Any,3}, 2, 2)) == Array{<:Any,3}
  end

  @testset "Set multiple parameters if they are unspecified" begin
    @test @inferred(specify_parameters(Array{Float32,3}, Float16, 2)) == Array{Float32,3}
    @test_broken @inferred((() -> specify_parameters(Array{Float32}, Float16, 2))()) ==
      Array{Float32,2}
    @test specify_parameters(Array{Float32}, Float16, 2) == Array{Float32,2}
    @test_broken @inferred(specify_parameters(Array{<:Any,3}, Float16, 2)) ==
      Array{Float16,3}
    @test specify_parameters(Array{<:Any,3}, Float16, 2) == Array{Float16,3}
    @test_broken @inferred((() -> specify_parameters(Array, Float16, 2))()) ==
      Array{Float16,2}
    @test (() -> specify_parameters(Array, Float16, 2))() == Array{Float16,2}
  end

  ## TODO Still working on the Default parameters system
  @testset "Default parameters" begin
    @test @inferred(default_parameter(Array{Float32,3}, Position(1))) == Float64
    @test @inferred(default_parameter(Array{Float32,3}, Position(2))) == 1
  end

  @testset "Set to the default parameter" begin
    @test @inferred(set_parameters(Array{Float32,3}, Position(1), DefaultParameter())) ==
      Array{Float64,3}
    @test @inferred(set_parameters(Array{Float32,3}, Position(2), DefaultParameter())) ==
      Array{Float32,1}
    @test @inferred(set_parameters(Array{Float32}, Position(1), DefaultParameter())) ==
      Array{Float64}
    @test @inferred(set_parameters(Array{Float32}, Position(2), DefaultParameter())) ==
      Array{Float32,1}
    @test @inferred(set_parameters(Array{<:Any,3}, Position(1), DefaultParameter())) ==
      Array{Float64,3}
    # TODO: Inferrence is broken for this case
    @test @inferred(Any, set_parameters(Array{<:Any,3}, Position(2), DefaultParameter())) ==
      Array{<:Any,1}
    @test @inferred(set_parameters(Array, Position(1), DefaultParameter())) ==
      Array{Float64}
    # TODO: Inferrence is broken for this case
    @test @inferred(Any, set_parameters(Array, Position(2), DefaultParameter())) ==
      Array{<:Any,1}
  end

  @testset "Set to the default parameters" begin
    @test @inferred(set_parameters(Array{Float32,3}, DefaultParameters())) ==
      Array{Float64,1}
    @test @inferred(set_parameters(Array{Float32}, DefaultParameters())) == Array{Float64,1}
    @test @inferred(set_parameters(Array{<:Any,3}, DefaultParameters())) == Array{Float64,1}
    @test @inferred(set_parameters(Array, DefaultParameters())) == Array{Float64,1}
  end

  @testset "Set to the default parameter if unspecified" begin
    @test @inferred(
      specify_parameters(Array{Float32,3}, Position(1), DefaultParameter())
    ) == Array{Float32,3}
    @test @inferred(
      specify_parameters(Array{Float32,3}, Position(2), DefaultParameter())
    ) == Array{Float32,3}
    @test @inferred(specify_parameters(Array{Float32}, Position(1), DefaultParameter())) ==
      Array{Float32}
    @test @inferred(specify_parameters(Array{Float32}, Position(2), DefaultParameter())) ==
      Array{Float32,1}
    @test @inferred(specify_parameters(Array{<:Any,3}, Position(1), DefaultParameter())) ==
      Array{Float64,3}
    @test @inferred(specify_parameters(Array{<:Any,3}, Position(2), DefaultParameter())) ==
      Array{<:Any,3}
    @test @inferred(specify_parameters(Array, Position(1), DefaultParameter())) ==
      Array{Float64}
    # TODO: Inferrence is broken for this case
    @test @inferred(Any, specify_parameters(Array, Position(2), DefaultParameter())) ==
      Array{<:Any,1}
  end

  @testset "Set to the default parameters if unspecified" begin
    @test @inferred(specify_parameters(Array{Float32,3}, DefaultParameters())) ==
      Array{Float32,3}
    @test @inferred(specify_parameters(Array{Float32}, DefaultParameters())) ==
      Array{Float32,1}
    @test @inferred(specify_parameters(Array{<:Any,3}, DefaultParameters())) ==
      Array{Float64,3}
    @test @inferred(specify_parameters(Array, DefaultParameters())) == Array{Float64,1}
  end

  @testset "Test unspecifing parameters" begin
    v = Vector{Float32}
    m = Matrix{Float64}
    a = Array{ComplexF32}
    val = Val{3}
    @test TypeParameterAccessors.unspecify_parameters(v) == Array
    @test TypeParameterAccessors.unspecify_parameters(m) == Array
    @test TypeParameterAccessors.unspecify_parameters(a) == Array
    @test TypeParameterAccessors.unspecify_parameters(val) == Val
  end
end
end
