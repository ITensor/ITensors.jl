struct MyType{T,N} end

struct MyTypeNamedParams{N,T,V} end

using NDTensors.TypeParameterAccessors: TypeParameterAccessors, Position
TypeParameterAccessors.position(::Type{<:MyTypeNamedParams}, ::typeof(eltype)) = Position(2)
TypeParameterAccessors.position(::Type{<:MyTypeNamedParams}, ::typeof(ndims)) = Position(1)
third_type(type::Type{<:MyTypeNamedParams}) = parameter(type, position(type, third_type))
function TypeParameterAccessors.position(::Type{<:MyTypeNamedParams}, ::typeof(third_type))
  return Position(3)
end

struct MyTypeDefaults{V,T,N} end

## TODO change this to use names
TypeParameterAccessors.default_parameters() = (Position(1), Position(2), Position(3))
TypeParameterAccessors.default_parameter(::Type{<:MyTypeDefaults}, ::Position{1}) = "3"
TypeParameterAccessors.default_parameter(::Type{<:MyTypeDefaults}, ::Position{2}) = Float32
TypeParameterAccessors.default_parameter(::Type{<:MyTypeDefaults}, ::Position{3}) = 2
