@eval module $(gensym())
using Test: @inferred, @test, @test_broken, @testset
using LinearAlgebra: Transpose
using NDTensors.TypeParameterAccessors
using NDTensors.TypeParameterAccessors: TypeParameter

function inferred(f::Function)
  return @inferred f()
end

@testset "Test NDTensors.TypeParameterAccessors" begin
  @testset "Get parameters" begin
    @test @inferred(parameters(Array{Float32,3})) == (Float32, 3)
    @test @inferred(parameters(Array{Float32,2}(undef, (2, 3)))) == (Float32, 2)
    @test inferred((() -> parameter(Array{Float32,3}, 1))) == Float32
    @test inferred((() -> parameter(Array{Float32,3}, 2))) == 3

    @test @inferred(parameter(Array{Float32,3}, Position(1))) == Float32
    @test @inferred(parameter(Array{Float32,3}, Position(2))) == 3
    @test nparameters(Array) == 2
    @test nparameters(Base.SubArray) == 5
  end

  @testset "Set parameter at position" begin
    @test inferred((() -> set_parameters(Array{Float32,3}, (Float16, 2)))) ==
      Array{Float16,2}
    @test inferred((() -> set_parameters(Array, (Float32, 3)))) == Array{Float32,3}
    @test @inferred(set_parameters(Array{Float32,2})) == Array{Float32,2}

    @test inferred((() -> set_parameter(Array{Float32,3}, 1, Float16))) == Array{Float16,3}
    @test inferred((() -> set_parameter(Array{Float32,3}, 2, 2))) == Array{Float32,2}
    @test inferred((() -> set_parameter(Array{Float32}, 1, Float16))) == Array{Float16}
    @test inferred((() -> set_parameter(Array{Float32}, 2, 2))) == Array{Float32,2}
    @test inferred((() -> set_parameter(Vector, 1, Float16))) == Array{Float16,1}
    @test inferred((() -> set_parameter(Vector, 2, 2))) == Matrix
    @test inferred((() -> set_parameter(Array, 1, Float16))) == Array{Float16}
    @test inferred((() -> set_parameter(Array, 2, 2))) == Matrix

    @test inferred((() -> set_parameter(Vector, Position(1), Float32))) == Vector{Float32}
    @test inferred((() -> set_parameter(Array{Float32}, Position(2), TypeParameter(2)))) ==
      Array{Float32,2}
    @test inferred((() -> set_parameter(Array{Float32}, Float16))) == Array{Float16}
    @test inferred((() -> set_parameter(Array{<:Any,2}, Position(2), TypeParameter(3)))) ==
      Array{<:Any,3}
  end

  @testset "Set ndim and eltype" begin
    @test @inferred((() -> set_ndims(Array{Float32,2}, 3))()) == Array{Float32,3}
    @test @inferred((() -> set_eltype(Array{Float32,2}, Float16))()) == Array{Float16,2}
    @test @inferred((() -> set_ndims(Array, 2))()) == Matrix
    @test @inferred(((() -> set_eltype(Array, Float32))())) == Array{Float32}

    ## TODO now I am more strict with `eltype` and `ndims` definition so These
    ## now don't work 
    m = Transpose(Matrix{Float32}(undef, (2, 3)))
    @test_broken (() -> set_eltype(typeof(m), Float16))() ==
      Transpose{Float16,Matrix{Float16}}
    @test_broken @inferred((() -> set_eltype(typeof(m), Float16))()) ==
      Transpose{Float16,Matrix{Float16}}
    @test_broken @inferred((() -> set_parameter(typeof(m), eltype, Float16))()) ==
      Transpose{Float16,Matrix{Float32}}

    @test ((() -> set_ndims(Array{<:Any,3}, 4))()) == Array{<:Any,4}
    @test ((() -> set_eltype(Array{<:Any,3}, Float16))()) == Array{Float16,3}
  end

  @testset "Set a parameter if it is unspecified" begin
    @test @inferred(specify_parameter(Array{Float32,3}, Position(1), Float16)) ==
      Array{Float32,3}
    @test @inferred(specify_parameter(Array{Float32,3}, Position(2), 2)) == Array{Float32,3}
    @test inferred((() -> specify_parameter(Array{Float32}, 1, Float16))) == Array{Float32}
    @test inferred((() -> specify_parameter(Array{Float32}, 2, 2))) == Array{Float32,2}
    @test inferred((() -> specify_parameter(Array, 1, Float16))) == Array{Float16}
    @test inferred((() -> specify_parameter(Array, 2, 2))) == Array{<:Any,2}
    @test inferred((() -> specify_parameter(Matrix, 1, Float32))) == Array{Float32,2}
    @test inferred((() -> specify_parameter(Matrix, 2, 2))) == Array{<:Any,2}
    @test inferred((() -> specify_parameter(Vector, 1, Float64))) == Array{Float64,1}
    @test inferred((() -> specify_parameter(Vector, 2, 2))) == Array{<:Any,1}

    @test inferred((() -> specify_parameter(Array, eltype, Float32))) == Array{Float32}
    @test inferred((() -> specify_parameter(Array, ndims, 2))) == Array{<:Any,2}

    @test specify_parameter(Array{<:Any,3}, 1, Float16) == Array{Float16,3}
    @test specify_parameter(Array{<:Any,3}, 2, 2) == Array{<:Any,3}
    ## TODO Do we want to support type stable this behavior, it does not seem suppported by julia.
    ## Though if you have `Matrix` or `Vector` the code is type stable
    @test @inferred(specify_parameter(Array{<:Any,3}, Position(1), Float16)) ==
      Array{Float16,3}
    @test @inferred(specify_parameter(Array{<:Any,3}, Position(2), 2)) == Array{<:Any,3}
  end

  @testset "Set multiple parameters if they are unspecified" begin
    @test inferred((() -> specify_parameter(specify_parameter(Array, 1, Float32), 2, 2))) ==
      Array{Float32,2}
    @test inferred((() -> specify_parameter(specify_parameter(Array, 2, 2), 1, Float32))) ==
      Array{Float32,2}

    @test inferred((
      () -> specify_parameter(specify_parameter(Matrix, 2, 3), 1, Float32)
    )) == Array{Float32,2}
    @test inferred((
      () -> specify_parameter(specify_parameter(Matrix, 1, Float32), 2, 3)
    )) == Array{Float32,2}
    @test inferred((
      () -> specify_parameter(specify_parameter(Vector, 1, Float32), 2, 3)
    )) == Array{Float32,1}
    @test inferred((
      () -> specify_parameter(specify_parameter(Array, eltype, Float32), ndims, 2)
    )) == Matrix{Float32}
    @test inferred((
      () -> specify_parameter(specify_parameter(Array, ndims, 2), eltype, Float32)
    )) == Matrix{Float32}

    @test inferred((() -> specify_parameters(Array, (1, 2), (Float32, 2)))) ==
      Array{Float32,2}
    @test inferred((() -> specify_parameters(Array, (2, 1), (2, Float32)))) ==
      Array{Float32,2}

    @test inferred((() -> specify_parameters(Matrix, (1, 2), (Float32, 3)))) ==
      Array{Float32,2}
    @test inferred((() -> specify_parameters(Matrix, (2, 1), (3, Float32)))) ==
      Array{Float32,2}

    @test inferred((() -> specify_parameters(Array, (eltype, ndims), (Float32, 2)))) ==
      Matrix{Float32}
    @test inferred((() -> specify_parameters(Array, (ndims, eltype), (2, Float32)))) ==
      Matrix{Float32}

    @test inferred((() -> specify_parameters(Array, (Position(1), ndims), (Float32, 2)))) ==
      Matrix{Float32}
    @test inferred((() -> specify_parameters(Array, (2, Position(1)), (2, Float32)))) ==
      Matrix{Float32}
    @test inferred((() -> specify_parameters(Array, (1, ndims), (Float32, 2)))) ==
      Matrix{Float32}

    @test @inferred((() -> specify_default_parameters(Array))()) == Vector{Float64}
    @test @inferred((() -> specify_default_parameters(Matrix))()) == Matrix{Float64}
    @test @inferred((() -> specify_default_parameters(Array{Float32}))()) == Vector{Float32}

    @test @inferred(
      (
        () -> TypeParameterAccessors.specify_parameters(
          Array, position_names(Array), (Float32, 2)
        )
      )()
    ) == Matrix{Float32}
  end

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
    @test @inferred(set_parameter(m, Position(1), Float32)) == MyType{Float32,'N'}
    @test set_parameter(m, 1, Float32) == MyType{Float32,'N'}
    @test set_parameter(m, Position(2), 2) == MyType{'T',2}
    @test set_parameter(m, 2, 2) == MyType{'T',2}
    @test set_parameters(m, (1, 2), (Float32, 2)) == MyType{Float32,2}
    @test specify_parameters(m, (1, 2), (Float32, 2)) == m
    @test specify_parameter(m, 1, Float32) == m
    @test specify_parameter(m, 2, Float32) == m
    @test specify_parameter(MyType{<:Any,2}, 1, Float32) == MyType{Float32,2}
    @test specify_parameter(MyType{<:Any,2}, 2, Float32) == MyType{<:Any,2}
  end

  @testset "Testing MyTypeNamedPositions" begin
    m = MyTypeNamedPositions{'T','N','D'}
    @test parameters(m) == ('T', 'N', 'D')
    @test parameter(m, 1) == 'T'
    @test parameter(m, Position(1)) == 'T'
    @test parameter(m, ndims) == 'T'

    @test @inferred(set_parameter(m, eltype, Float32)) ==
      MyTypeNamedPositions{'T',Float32,'D'}
    @test set_parameter(m, third, 'S') == MyTypeNamedPositions{'T','N','S'}
    @test specify_parameters(m, (ndims, third), (2, 's')) == m
    @test specify_parameters(MyTypeNamedPositions, (ndims, third), (2, 's')) ==
      MyTypeNamedPositions{2,<:Any,'s'}
  end

  @testset "Testing MyTypeDefaultPositions" begin
    m = MyTypeDefaultPositions{'T','N','D'}
    @test parameters(m) == ('T', 'N', 'D')
    @test parameter(m, 1) == 'T'
    @test parameter(m, Position(1)) == 'T'
    #@test @error parameter(m, ndims) == 'T'

    @test @inferred(set_parameter(m, Position(2), Float32)) ==
      MyTypeDefaultPositions{'T',Float32,'D'}
    @test set_parameter(m, 3, 'S') == MyTypeDefaultPositions{'T','N','S'}
    @test specify_default_parameters(MyTypeDefaultPositions) ==
      MyTypeDefaultPositions{2,Float16,'S'}
    @test specify_default_parameters(MyTypeDefaultPositions, (Position(2), Position(1))) ==
      MyTypeDefaultPositions{2,Float16}
    @test specify_default_parameters(
      MyTypeDefaultPositions{<:Any,<:Any,'L'}, (Position(2), Position(1))
    ) == MyTypeDefaultPositions{2,Float16,'L'}
    @test set_default_parameters(m) == MyTypeDefaultPositions{2,Float16,'S'}
    @test set_default_parameter(m, Position(3)) == MyTypeDefaultPositions{'T','N','S'}
  end

  @testset "Testing MyTypeDefaultNamedPosition" begin
    m = MyTypeDefaultNamedPosition{1,2,3}
    @test parameters(m) == (1, 2, 3)
    @test parameter(m, 1) == 1
    @test parameter(m, Position(1)) == 1
    @test parameter(m, ndims) == 1

    @test @inferred(set_parameter(m, eltype, Float32)) ==
      MyTypeDefaultNamedPosition{1,Float32,3}
    @test set_parameter(m, third, 'S') == MyTypeDefaultNamedPosition{1,2,'S'}
    @test set_parameters(m, (third, ndims, eltype), ('S', 5, Float32)) ==
      MyTypeDefaultNamedPosition{5,Float32,'S'}
    @test specify_parameter(m, third, Float32) == m
    @test specify_parameter(MyTypeDefaultNamedPosition, eltype, Float32) ==
      MyTypeDefaultNamedPosition{<:Any,Float32}

    @test specify_default_parameter(MyTypeDefaultNamedPosition, eltype) ==
      MyTypeDefaultNamedPosition{<:Any,Float32}
    @test specify_default_parameter(MyTypeDefaultNamedPosition, third) ==
      MyTypeDefaultNamedPosition{<:Any,<:Any,'P'}
    @test specify_default_parameters(MyTypeDefaultNamedPosition, (ndims, third)) ==
      MyTypeDefaultNamedPosition{3,<:Any,'P'}

    TypeParameterAccessors.position_name(
      ::Type{<:MyTypeDefaultNamedPosition}, ::Position{1}
    ) = ndims
    TypeParameterAccessors.position_name(
      ::Type{<:MyTypeDefaultNamedPosition}, ::Position{2}
    ) = eltype
    TypeParameterAccessors.position_name(
      ::Type{<:MyTypeDefaultNamedPosition}, ::Position{3}
    ) = third
    @test specify_default_parameters(MyTypeDefaultNamedPosition) ==
      MyTypeDefaultNamedPosition{3,Float32,'P'}
    @test set_default_parameters(m) == MyTypeDefaultNamedPosition{3,Float32,'P'}
  end
end
end
