@eval module $(gensym())
using Test: @testset
using LinearAlgebra:
  Adjoint,
  Diagonal,
  Hermitian,
  LowerTriangular,
  Symmetric,
  Transpose,
  UnitLowerTriangular,
  UnitUpperTriangular,
  UpperTriangular
using NDTensors.TypeParameterAccessors:
  TypeParameter,
  is_wrapped_array,
  parenttype,
  set_eltype,
  set_ndims,
  set_parenttype,
  unwrap_array_type
using StridedViews: StridedView
include("utils/test_inferred.jl")
@testset "TypeParameterAccessors wrapper types" begin
  @testset "Array" begin
    array = randn(2, 2)
    array_type = typeof(array)
    @test_inferred parenttype(Matrix) == Matrix
    @test_inferred is_wrapped_array(array) == false
    @test_inferred parenttype(array) == array_type
    @test_inferred set_eltype(array, Float32) isa Matrix{Float32}
    @test_inferred set_eltype(array, Float32) ≈ array
    @test_inferred set_eltype(Array{<:Any,2}, Float64) == Matrix{Float64}
    @test_inferred set_ndims(Array{Float64}, 2) == Matrix{Float64} wrapped = true
    @test_inferred set_ndims(Array{Float64}, TypeParameter(2)) == Matrix{Float64}
    @test_inferred unwrap_array_type(array_type) == array_type
  end
  @testset "Base AbstractArray wrappers" begin
    array = randn(2, 2)
    for wrapped_array in (
      Base.ReshapedArray(array, (2, 2), ()),
      SubArray(randn(2, 2), (:, :)),
      PermutedDimsArray(randn(2, 2), (1, 2)),
    )
      wrapped_array_type = typeof(wrapped_array)
      @test parenttype(wrapped_array) == Matrix{Float64}
      @test is_wrapped_array(wrapped_array)
      @test unwrap_array_type(wrapped_array_type) == Matrix{Float64}
      # TODO: Test `set_eltype`.
    end
  end
  @testset "LinearAlgebra wrappers" begin
    for wrapper in (
      Transpose,
      Adjoint,
      Symmetric,
      Hermitian,
      UpperTriangular,
      LowerTriangular,
      UnitUpperTriangular,
      UnitLowerTriangular,
    )
      array = randn(2, 2)
      wrapped_array = wrapper(array)
      wrapped_array_type = typeof(wrapped_array)
      @test_inferred is_wrapped_array(wrapped_array) == true
      @test_inferred parenttype(wrapped_array) == Matrix{Float64}
      @test_inferred unwrap_array_type(wrapped_array_type) == Matrix{Float64}
      @test_inferred set_parenttype(wrapped_array_type, wrapped_array_type) ==
        wrapper{eltype(wrapped_array_type),wrapped_array_type}
      @test_inferred set_eltype(wrapped_array_type, Float32) ==
        wrapper{Float32,Matrix{Float32}}
      if wrapper ∉ (UnitUpperTriangular, UnitLowerTriangular)
        @test_inferred set_eltype(wrapped_array, Float32) isa
          wrapper{Float32,Matrix{Float32}}
      end
    end
  end
  @testset "LinearAlgebra Diagonal wrapper" begin
    array = randn(2, 2)
    wrapped_array = Diagonal(array)
    wrapped_array_type = typeof(wrapped_array)
    @test_inferred is_wrapped_array(wrapped_array) == true
    @test_inferred parenttype(wrapped_array) == Vector{Float64}
    @test_inferred unwrap_array_type(wrapped_array_type) == Vector{Float64}
    @test_inferred set_eltype(wrapped_array_type, Float32) ==
      Diagonal{Float32,Vector{Float32}}
    @test_inferred set_eltype(wrapped_array, Float32) isa Diagonal{Float32,Vector{Float32}}
  end
  @testset "LinearAlgebra nested wrappers" begin
    array = randn(2, 2)
    wrapped_array = view(reshape(transpose(array), 4), 1:2)
    wrapped_array_type = typeof(wrapped_array)
    @test_inferred is_wrapped_array(wrapped_array) == true
    @test_inferred parenttype(wrapped_array) <:
      Base.ReshapedArray{Float64,1,Transpose{Float64,Matrix{Float64}}}
    @test_inferred unwrap_array_type(array) == Matrix{Float64}
  end
  @testset "StridedView" begin
    array = randn(2, 2)
    wrapped_array = StridedView(randn(2, 2))
    wrapped_array_type = typeof(wrapped_array)
    @test_inferred is_wrapped_array(wrapped_array) == true
    @test_inferred parenttype(wrapped_array) == Matrix{Float64}
    @test_inferred unwrap_array_type(wrapped_array_type) == Matrix{Float64}
  end
end
end
