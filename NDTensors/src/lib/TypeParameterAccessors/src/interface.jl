"""
  position(type::Type, position_name)::Position

An optional interface function. Defining this allows accessing a parameter
at the defined position using the `position_name`.

For example, defining `TypeParameterAccessors.position(::Type{<:MyType}, ::typeof(eltype)) = Position(1)`
allows accessing the first type parameter with `type_parameter(MyType(...), eltype)`,
in addition to the standard `type_parameter(MyType(...), 1)` or `type_parameter(MyType(...), Position(1))`.
"""
function position end

"""
  default_parameters(type::Type)::Tuple

An optional interface function. Defining this allows filling type parameters
of the specified type with default values.

This function should output a Tuple of the default values, with exactly
one for each type parameter slot of the type.
"""
function default_type_parameters end
