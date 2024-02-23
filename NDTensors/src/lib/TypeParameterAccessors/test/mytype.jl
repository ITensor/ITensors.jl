using .TypeParameterAccessors: TypeParameterAccessors, Position
struct MyType{T,N} end

struct MyTypeNamedPositions{N,T,V} end

third(type::Type{<:MyTypeNamedPositions}) = parameter(type, third)

function TypeParameterAccessors.position(::Type{<:MyTypeNamedPositions}, ::typeof(ndims))
  return Position(1)
end
function TypeParameterAccessors.position(::Type{<:MyTypeNamedPositions}, ::typeof(eltype))
  return Position(2)
end
function TypeParameterAccessors.position(::Type{<:MyTypeNamedPositions}, ::typeof(third))
  return Position(3)
end

struct MyTypeDefaultPositions{N,T,V} end

function TypeParameterAccessors.default_parameter(
  ::Type{<:MyTypeDefaultPositions}, ::Position{1}
)
  return 2
end
function TypeParameterAccessors.default_parameter(
  ::Type{<:MyTypeDefaultPositions}, ::Position{2}
)
  return Float16
end
function TypeParameterAccessors.default_parameter(
  ::Type{<:MyTypeDefaultPositions}, ::Position{3}
)
  return 'S'
end

struct MyTypeDefaultNamedPosition{V,T,N} end
third(::Type{<:MyTypeDefaultNamedPosition}) = parameter(type, third)

function TypeParameterAccessors.position(
  ::Type{<:MyTypeDefaultNamedPosition}, ::typeof(ndims)
)
  return Position(1)
end
function TypeParameterAccessors.position(
  ::Type{<:MyTypeDefaultNamedPosition}, ::typeof(eltype)
)
  return Position(2)
end
function TypeParameterAccessors.position(
  ::Type{<:MyTypeDefaultNamedPosition}, ::typeof(third)
)
  return Position(3)
end

function TypeParameterAccessors.default_parameter(
  ::Type{<:MyTypeDefaultNamedPosition}, ::typeof(ndims)
)
  return 3
end
function TypeParameterAccessors.default_parameter(
  ::Type{<:MyTypeDefaultNamedPosition}, ::typeof(eltype)
)
  return Float32
end
function TypeParameterAccessors.default_parameter(
  ::Type{<:MyTypeDefaultNamedPosition}, ::typeof(third)
)
  return 'P'
end
