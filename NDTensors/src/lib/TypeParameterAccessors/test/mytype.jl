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
function TypeParameterAccessors.default_parameter(
  ::Type{<:MyTypeDefaults}, ::typeof(third_type)
)
  return "3"
end
function TypeParameterAccessors.default_parameter(
  ::Type{<:MyTypeDefaults}, ::typeof(eltype)
)
  return Float32
end
TypeParameterAccessors.default_parameter(::Type{<:MyTypeDefaults}, ::typeof(ndims)) = 2

TypeParameterAccessors.parameter_name(::Type{<:MyTypeDefaults}, ::Position{1}) = third_type
TypeParameterAccessors.parameter_name(::Type{<:MyTypeDefaults}, ::Position{2}) = eltype
TypeParameterAccessors.parameter_name(::Type{<:MyTypeDefaults}, ::Position{3}) = ndims

function TypeParameterAccessors.position(::Type{<:MyTypeDefaults}, ::typeof(third_type))
  return Position(1)
end
TypeParameterAccessors.position(::Type{<:MyTypeDefaults}, ::typeof(eltype)) = Position(2)
TypeParameterAccessors.position(::Type{<:MyTypeDefaults}, ::typeof(ndims)) = Position(3)
