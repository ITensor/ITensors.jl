@eval module $(gensym())
using Test: @inferred, @test, @test_broken, @testset
using NDTensors.TypeParameterAccessors
using NDTensors.TypeParameterAccessors: specify_parameters, ndims
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
    @test @inferred((() -> set_eltype(typeof(m), Float16))()) ==
      Transpose{Float16,Matrix{Float16}}
    @test @inferred((() -> set_parameter(typeof(m), eltype, Float16))()) ==
      Transpose{Float16,Matrix{Float32}}
    ## TODO This code does not infer the correct type but there aren't any allocations
    ## When it is called so I believe its actually working properly
    ## In a wrapped function on the command line this does however show the correct
    ## value when using code_warntype.
    @test ((() -> set_ndims(Array{<:Any,3}, 4))()) == Array{<:Any,4}
    @test ((() -> set_eltype(Array{<:Any,3}, Float16))()) == Array{Float16,3}
  end

  @testset "Set a parameter if it is unspecified" begin
    @test @inferred(specify_parameter(Array{Float32,3}, 1, Float16)) == Array{Float32,3}
    @test @inferred(specify_parameter(Array{Float32,3}, 2, 2)) == Array{Float32,3}
    @test @inferred((() -> specify_parameter(Array{Float32}, 1, Float16))()) ==
      Array{Float32}
    @test @inferred((() -> specify_parameter(Array{Float32}, 2, 2))()) == Array{Float32,2}
    @test @inferred((() -> specify_parameter(Array, 1, Float16))()) == Array{Float16}
    @test @inferred((() -> specify_parameter(Array, 2, 2))()) == Array{<:Any,2}
    @test @inferred((() -> specify_parameter(Matrix, 1, Float32))()) == Array{Float32,2}
    @test @inferred((() -> specify_parameter(Matrix, 2, 2))()) == Array{<:Any,2}
    @test @inferred((() -> specify_parameter(Vector, 1, Float64))()) == Array{Float64,1}
    @test @inferred((() -> specify_parameter(Vector, 2, 2))()) == Array{<:Any,1}

    @test @inferred((() -> specify_parameter(Array, eltype, Float32))()) == Array{Float32}
    @test @inferred((() -> specify_parameter(Array, ndims, 2))()) == Array{<:Any,2}

    @test specify_parameter(Array{<:Any,3}, 1, Float16) == Array{Float16,3}
    @test specify_parameter(Array{<:Any,3}, 2, 2) == Array{<:Any,3}
    ## TODO Do we want to support type stable this behavior, it does not seem suppported by julia.
    ## Though if you have `Matrix` or `Vector` the code is type stable
    @test_broken @inferred(specify_parameter(Array{<:Any,3}, 1, Float16)) ==
      Array{Float16,3}
    @test_broken @inferred(specify_parameters(Array{<:Any,3}, 2, 2)) == Array{<:Any,3}
  end

  @testset "Set multiple parameters if they are unspecified" begin
    @test @inferred(
      (() -> specify_parameter(specify_parameter(Array, 1, Float32), 2, 2))()
    ) == Array{Float32,2}
    @test @inferred(
      (() -> specify_parameter(specify_parameter(Array, 2, 2), 1, Float32))()
    ) == Array{Float32,2}

    @test @inferred(
      (() -> specify_parameter(specify_parameter(Matrix, 2, 3), 1, Float32))()
    ) == Array{Float32,2}
    @test @inferred(
      (() -> specify_parameter(specify_parameter(Matrix, 1, Float32), 2, 3))()
    ) == Array{Float32,2}
    @test @inferred(
      (() -> specify_parameter(specify_parameter(Vector, 1, Float32), 2, 3))()
    ) == Array{Float32,1}
    @test @inferred(
      (() -> specify_parameter(specify_parameter(Array, eltype, Float32), ndims, 2))()
    ) == Matrix{Float32}
    @test @inferred(
      (() -> specify_parameter(specify_parameter(Array, ndims, 2), eltype, Float32))()
    ) == Matrix{Float32}

    @test @inferred((() -> specify_default_parameters(Array))()) == Vector{Float64}
    @test @inferred((() -> specify_default_parameters(Matrix))()) == Matrix{Float64}
    @test @inferred((() -> specify_default_parameters(Array{Float32}))()) == Vector{Float32}

    @test @inferred(
      (
        () -> TypeParameterAccessors.specify_parameters(
          Array, parameter_names(Array), (Float32, 2)
        )
      )()
    ) == Matrix{Float32}
  end

  ## TODO Add more tests for the default parameters like
  #specify_parameters(Array, (eltype, 2), (Float32, 2)) == Matrix{Float32}
  ## also add tests for set_parameter in the same way
  # @testset "Default parameters" begin
  #   @test @inferred(default_parameter(Array{Float32,3}, Position(1))) == Float64
  #   @test @inferred(default_parameter(Array{Float32,3}, Position(2))) == 1
  # end

  @testset "Test unspecifying parameters" begin
    v = Vector{Float32}
    m = Matrix{Float64}
    a = Array{ComplexF32}
    val = Val{3}
    @test TypeParameterAccessors.unspecify_parameters(v) == Array
    @test TypeParameterAccessors.unspecify_parameters(m) == Array
    @test TypeParameterAccessors.unspecify_parameters(a) == Array
    @test TypeParameterAccessors.unspecify_parameters(val) == Val
  end

  include("mytype.jl")
  @testset "Testing MyType" begin
    m = MyType{'T','N'}
    @test parameters(m) == ('T', 'N')
    @test parameter(m, 1) == 'T'
    @test parameter(m, Position(1)) == 'T'
    @test_broken @inferred((() -> set_parameter(m, 1, Float32))()) == MyType{Float32,'N'}
    @test_broken @inferred((() -> set_parameter(m, 2, 2))()) == MyType{'T',2}
    @test_broken @inferred((() -> set_parameter(m, Position(1), Float32))()) ==
      MyType{Float32,'N'}
    @test_broken @inferred((() -> set_parameter(m, Position(2), TypeParameter(2)))()) ==
      MyType{'T',2}
    @test ((() -> set_parameter(m, 1, Float32))()) == MyType{Float32,'N'}
    @test ((() -> set_parameter(m, 2, 2))()) == MyType{'T',2}
    @test ((() -> set_parameter(m, Position(1), Float32))()) == MyType{Float32,'N'}
    @test ((() -> set_parameter(m, Position(2), TypeParameter(2)))()) == MyType{'T',2}
    @test set_parameters(MyType, (1, 2), (Float32, 2)) == MyType{Float32,2}
    @test @inferred((() -> set_parameters(MyType, (1, 2), (Float32, 2)))()) ==
      MyType{Float32,2}
  end

  ## TODO working here still
  @testset "Testing MyTypeNamedParams" begin
    m = MyType{'T','N'}
    @test parameters(m) == ('T', 'N')
    @test parameter(m, 1) == 'T'
    @test parameter(m, Position(1)) == 'T'
    @test parameter(m, Position(1)) == 'T'
    @test_broken @inferred((() -> set_parameter(m, 1, Float32))()) == MyType{Float32,'N'}
    @test_broken @inferred((() -> set_parameter(m, 2, 2))()) == MyType{'T',2}
    @test_broken @inferred((() -> set_parameter(m, Position(1), Float32))()) ==
      MyType{Float32,'N'}
    @test_broken @inferred((() -> set_parameter(m, Position(2), TypeParameter(2)))()) ==
      MyType{'T',2}
    @test ((() -> set_parameter(m, 1, Float32))()) == MyType{Float32,'N'}
    @test ((() -> set_parameter(m, 2, 2))()) == MyType{'T',2}
    @test ((() -> set_parameter(m, Position(1), Float32))()) == MyType{Float32,'N'}
    @test ((() -> set_parameter(m, Position(2), TypeParameter(2)))()) == MyType{'T',2}

    @test set_parameters(MyType, (1, 2), (Float32, 2)) == MyType{Float32,2}
    @test @inferred((() -> set_parameters(MyType, (1, 2), (Float32, 2)))()) ==
      MyType{Float32,2}
  end
end
end
