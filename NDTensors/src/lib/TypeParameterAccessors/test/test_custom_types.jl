@eval module $(gensym())
using Test: @testset
@testset "TypeParameterAccessors custom types" begin
  @eval module $(gensym())
    using Test: @testset
    using NDTensors.TypeParameterAccessors:
      TypeParameterAccessors,
      Position,
      default_type_parameter,
      default_type_parameters,
      set_default_type_parameter,
      set_default_type_parameters,
      set_type_parameter,
      set_type_parameters,
      specify_default_type_parameter,
      specify_default_type_parameters,
      specify_type_parameter,
      specify_type_parameters,
      type_parameter,
      type_parameters,
      unspecify_type_parameter,
      unspecify_type_parameters
    include("utils/test_inferred.jl")
    @testset "TypeParameterAccessors, named positions and defaults" begin
      struct MyType{P1,P2} end
      TypeParameterAccessors.default_type_parameters(::Type{<:MyType}) = (:P1, :P2)

      @test_inferred default_type_parameter(MyType, 1) == :P1 wrapped = true
      @test_inferred default_type_parameter(MyType, Position(1)) == :P1
      @test_inferred default_type_parameter(MyType, 2) == :P2 wrapped = true
      @test_inferred default_type_parameter(MyType, Position(2)) == :P2
      @test_inferred default_type_parameter(MyType{<:Any,2}, Position(1)) == :P1
      @test_inferred default_type_parameter(MyType{<:Any,2}, Position(2)) == :P2
      @test_inferred default_type_parameters(MyType{<:Any,2}) == (:P1, :P2)
      @test_inferred default_type_parameters(MyType) == (:P1, :P2)
      # TODO: These don't infer, need to investigate.
      @test_inferred default_type_parameter(MyType{<:Any,2}, 1) == :P1 inferred = false
      @test_inferred default_type_parameter(MyType{<:Any,2}, 2) == :P2 inferred = false

      @test_inferred set_default_type_parameter(MyType{1,2}, 1) == MyType{:P1,2} wrapped =
        true
      @test_inferred set_default_type_parameter(MyType{1,2}, Position(1)) == MyType{:P1,2}
      @test_inferred set_default_type_parameter(MyType{<:Any,2}, Position(1)) ==
        MyType{:P1,2}
      @test_inferred set_default_type_parameter(MyType{<:Any,2}, Position(2)) ==
        MyType{<:Any,:P2}
      @test_inferred set_default_type_parameters(MyType{<:Any,2}) == MyType{:P1,:P2}
      # TODO: These don't infer, need to investigate.
      @test_inferred set_default_type_parameter(MyType{<:Any,2}, 1) == MyType{:P1,2} inferred =
        false
      @test_inferred set_default_type_parameter(MyType{<:Any,2}, 2) == MyType{<:Any,:P2} inferred =
        false

      @test_inferred specify_default_type_parameter(MyType{<:Any,2}, Position(1)) ==
        MyType{:P1,2}
      @test_inferred specify_default_type_parameters(MyType{<:Any,2}) == MyType{:P1,2}
      @test_inferred specify_default_type_parameter(MyType{<:Any,2}, Position(2)) ==
        MyType{<:Any,2}
      @test_inferred specify_default_type_parameters(MyType) == MyType{:P1,:P2}
      # TODO: These don't infer, need to investigate.
      @test_inferred specify_default_type_parameter(MyType{<:Any,2}, 2) == MyType{<:Any,2} inferred =
        false

      # Named positions
      function p1 end
      function p2 end
      ## TODO remove TypeParameterAccessors when SetParameters is removed
      TypeParameterAccessors.position(::Type{<:MyType}, ::typeof(p1)) = Position(1)
      TypeParameterAccessors.position(::Type{<:MyType}, ::typeof(p2)) = Position(2)

      @test_inferred type_parameter(MyType{:p1}, p1) == :p1
      @test_inferred type_parameter(MyType{<:Any,:p2}, p2) == :p2
      @test_inferred default_type_parameter(MyType, p1) == :P1
      @test_inferred default_type_parameter(MyType, p2) == :P2
    end
  end
end
end
