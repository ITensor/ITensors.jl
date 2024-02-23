# Used for named positions.
"""
  position(type::Type, pos)

An optional interface function. You can define tag a parameters order with a funciton name
for example `position(::Type{<:MyType}, eltype) = Position(4)`
"""
function position(type::Type, pos)
  return error(
    "Type parameter position not defined for type `$(type)` and position name `$(pos)`."
  )
end

## position_name
"""
  position(type::Type, pos)

An optional interface function. You can define a mapping from parameter to the order in
the parameter list for example `position_name(::Type{<:MyType}, ::Position{4}) = eltype`
"""
function position_name(type::Type, pos::Position)
  return error(
    "The type parameter position `$(pos)` of type `$(type)` does not have a name defined for it.",
  )
end

## default_parameter
"""
  default_parameter(type::Type, pos::Position)

An optional interface function. To opt into this function you can either redefine this 
funciton on your type `default_parameter(type::Type{<:MyType}, pos::Position{4}) = Float32`
"""
function default_parameter(type::Type, pos::Position)
  return default_parameter(type, position_name(type, pos))
end

"""
  default_parameter(type::Type, pos)

An optional interface function. To opt into this function you can either redefine this 
function based on a positioned name `default_parameter(::Type{<:MyType}, ::typeof(eltype)) = Float32` 
This also requires that one define the mapping from function name to position 
`position(::Type{<:MyType}, ::typeof(eltype)) = Position(4)`.
"""
default_parameter(type::Type, pos) = default_parameter(type, position(type, pos))