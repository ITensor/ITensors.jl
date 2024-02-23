using .TypeParameterAccessors: TypeParameterAccessors, Position
struct MyType{T,N} end

struct MyTypeNamedPositions{N,T,V} end

  third(type::Type{<:MyTypeNamedPositions}) = parameter(type, third)

  TypeParameterAccessors.position(::Type{<:MyTypeNamedPositions}, ::typeof(ndims)) = Position(1)
  TypeParameterAccessors.position(::Type{<:MyTypeNamedPositions}, ::typeof(eltype)) = Position(2)
  TypeParameterAccessors.position(::Type{<:MyTypeNamedPositions}, ::typeof(third)) = Position(3)

struct MyTypeDefaultPositions{N,T,V} end

  TypeParameterAccessors.default_parameter(::Type{<:MyTypeDefaultPositions}, ::Position{1}) = 2
  TypeParameterAccessors.default_parameter(::Type{<:MyTypeDefaultPositions}, ::Position{2}) = Float16
  TypeParameterAccessors.default_parameter(::Type{<:MyTypeDefaultPositions}, ::Position{3}) = 'S'


struct MyTypeDefaultNamedPosition{V,T,N} end
  third(::Type{<:MyTypeDefaultNamedPosition}) = parameter(type, third)

  TypeParameterAccessors.position(::Type{<:MyTypeDefaultNamedPosition}, ::typeof(ndims)) = Position(1)
  TypeParameterAccessors.position(::Type{<:MyTypeDefaultNamedPosition}, ::typeof(eltype)) = Position(2)
  TypeParameterAccessors.position(::Type{<:MyTypeDefaultNamedPosition}, ::typeof(third)) = Position(3)

  TypeParameterAccessors.default_parameter(::Type{<:MyTypeDefaultNamedPosition}, ::typeof(ndims)) = 3
  TypeParameterAccessors.default_parameter(::Type{<:MyTypeDefaultNamedPosition}, ::typeof(eltype)) = Float32
  TypeParameterAccessors.default_parameter(::Type{<:MyTypeDefaultNamedPosition}, ::typeof(third)) = 'P'

