"""
Set the first type parameter.
"""
function set_parameter(type::Type, parameter)
  return set_parameter(type, Position(1), parameter)
end

"""
Set multiple type parameters starting from `position`.
"""
set_parameters(type::Type, position::Position, parameters...) = set_parameters(set_parameters, type, position, parameters...)

# Generic case of 1 parameter. This is the base case, and types should overload:
# ```julia
# set_parameter(type::Type, position::Position, parameter)
# ```
set_parameters(type::Type, position::Position, parameter) = set_parameter(type, position, parameter)

# Generic case of no parameters
set_parameters(type::Type, position::Position) = type

"""
Set multiple type parameters.
"""
set_parameters(type::Type, parameters...) = set_parameters(type, Position(1), parameters...)
