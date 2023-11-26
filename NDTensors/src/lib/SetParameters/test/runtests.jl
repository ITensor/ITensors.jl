using Test
using NDTensors.SetParameters

@testset "Test NDTensors.SetParameters" begin
  @testset "Get parameters" begin
    @test @inferred(get_parameters(Array{Float32,3})) == (Float32, 3)
    @test @inferred(get_parameter(Array{Float32,3}, Position(1))) == Float32
    @test @inferred(get_parameter(Array{Float32,3}, Position(2))) == 3
  end

  @testset "Set parameter at position" begin
    @test @inferred(set_parameters(Array{Float32,3}, Float16)) == Array{Float16,3}
    @test @inferred(set_parameters(Array{Float32,3}, Position(1), Float16)) ==
      Array{Float16,3}
    @test @inferred((() -> set_parameters(Array{Float32,3}, Position(2), 2))()) ==
      Array{Float32,2}
    @test @inferred(set_parameters(Array{Float32}, Float16)) == Array{Float16}
    @test @inferred(set_parameters(Array{Float32}, Position(1), Float16)) == Array{Float16}
    @test @inferred((() -> set_parameters(Array{Float32}, Position(2), 2))()) ==
      Array{Float32,2}
    @test @inferred(set_parameters(Array{<:Any,3}, Float16)) == Array{Float16,3}
    @test @inferred(set_parameters(Array{<:Any,3}, Position(1), Float16)) ==
      Array{Float16,3}
    # TODO: Inferrence is broken for this case
    @test @inferred(Any, (() -> set_parameters(Array{<:Any,3}, Position(2), 2))()) ==
      Array{<:Any,2}
    @test @inferred(set_parameters(Array, Float16)) == Array{Float16}
    @test @inferred(set_parameters(Array, Position(1), Float16)) == Array{Float16}
    @test @inferred((() -> set_parameters(Array, Position(2), 2))()) == Array{<:Any,2}
  end

  @testset "Set multiple parameters" begin
    @test @inferred((() -> set_parameters(Array{<:Any,3}, Float16, 2))()) ==
      Array{Float16,2}
    @test @inferred((() -> set_parameters(Array{<:Any,3}, Position(1), Float16, 2))()) ==
      Array{Float16,2}
    @test @inferred(set_parameters(Array{<:Any,3}, Float16)) == Array{Float16,3}
    @test @inferred(set_parameters(Array{<:Any,3}, Position(1), Float16)) ==
      Array{Float16,3}
    @test @inferred(set_parameters(Array{<:Any,3})) == Array{<:Any,3}
    @test @inferred(set_parameters(Array{<:Any,3}, Position(1))) == Array{<:Any,3}
    # TODO: Inferrence is broken for this case
    @test @inferred(Any, (() -> set_parameters(Array{<:Any,3}, Position(2), 2))()) ==
      Array{<:Any,2}
    @test @inferred(set_parameters(Array{<:Any,3}, Position(2))) == Array{<:Any,3}
  end

  @testset "Set a parameter if it is unspecified" begin
    @test @inferred(set_unspecified_parameters(Array{Float32,3}, Float16)) ==
      Array{Float32,3}
    @test @inferred(set_unspecified_parameters(Array{Float32,3}, Position(1), Float16)) ==
      Array{Float32,3}
    @test @inferred(set_unspecified_parameters(Array{Float32,3}, Position(2), 2)) ==
      Array{Float32,3}
    @test @inferred(set_unspecified_parameters(Array{Float32}, Float16)) == Array{Float32}
    @test @inferred(set_unspecified_parameters(Array{Float32}, Position(1), Float16)) ==
      Array{Float32}
    @test @inferred((() -> set_unspecified_parameters(Array{Float32}, Position(2), 2))()) ==
      Array{Float32,2}
    @test @inferred(set_unspecified_parameters(Array{<:Any,3}, Float16)) == Array{Float16,3}
    @test @inferred(set_unspecified_parameters(Array{<:Any,3}, Position(1), Float16)) ==
      Array{Float16,3}
    @test @inferred(set_unspecified_parameters(Array{<:Any,3}, Position(2), 2)) ==
      Array{<:Any,3}
    @test @inferred(set_unspecified_parameters(Array, Float16)) == Array{Float16}
    @test @inferred(set_unspecified_parameters(Array, Position(1), Float16)) ==
      Array{Float16}
    @test @inferred((() -> set_unspecified_parameters(Array, Position(2), 2))()) ==
      Array{<:Any,2}
  end

  @testset "Set multiple parameters if they are unspecified" begin
    @test @inferred(set_unspecified_parameters(Array{Float32,3}, Float16, 2)) ==
      Array{Float32,3}
    @test @inferred((() -> set_unspecified_parameters(Array{Float32}, Float16, 2))()) ==
      Array{Float32,2}
    @test @inferred(set_unspecified_parameters(Array{<:Any,3}, Float16, 2)) ==
      Array{Float16,3}
    @test @inferred((() -> set_unspecified_parameters(Array, Float16, 2))()) ==
      Array{Float16,2}
  end

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
      set_unspecified_parameters(Array{Float32,3}, Position(1), DefaultParameter())
    ) == Array{Float32,3}
    @test @inferred(
      set_unspecified_parameters(Array{Float32,3}, Position(2), DefaultParameter())
    ) == Array{Float32,3}
    @test @inferred(
      set_unspecified_parameters(Array{Float32}, Position(1), DefaultParameter())
    ) == Array{Float32}
    @test @inferred(
      set_unspecified_parameters(Array{Float32}, Position(2), DefaultParameter())
    ) == Array{Float32,1}
    @test @inferred(
      set_unspecified_parameters(Array{<:Any,3}, Position(1), DefaultParameter())
    ) == Array{Float64,3}
    @test @inferred(
      set_unspecified_parameters(Array{<:Any,3}, Position(2), DefaultParameter())
    ) == Array{<:Any,3}
    @test @inferred(set_unspecified_parameters(Array, Position(1), DefaultParameter())) ==
      Array{Float64}
    # TODO: Inferrence is broken for this case
    @test @inferred(
      Any, set_unspecified_parameters(Array, Position(2), DefaultParameter())
    ) == Array{<:Any,1}
  end

  @testset "Set to the default parameters if unspecified" begin
    @test @inferred(set_unspecified_parameters(Array{Float32,3}, DefaultParameters())) ==
      Array{Float32,3}
    @test @inferred(set_unspecified_parameters(Array{Float32}, DefaultParameters())) ==
      Array{Float32,1}
    @test @inferred(set_unspecified_parameters(Array{<:Any,3}, DefaultParameters())) ==
      Array{Float64,3}
    @test @inferred(set_unspecified_parameters(Array, DefaultParameters())) ==
      Array{Float64,1}
  end
end
