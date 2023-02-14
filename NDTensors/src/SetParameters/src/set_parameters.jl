"""
Set the first type parameter.
"""
function set_parameter(type::Type, parameter)
  return set_parameter(type, Position(1), parameter)
end

"""
Set multiple type parameters starting from `position`.
"""
function set_parameters(type::Type, position::Position, parameters...)
  return set_parameters(set_parameters, type, position, parameters...)
end

# Generic case of 1 parameter. This is the base case, and types should overload:
# ```julia
# set_parameter(type::Type, position::Position, parameter)
# ```
function set_parameters(type::Type, position::Position, parameter)
  return set_parameter(type, position, parameter)
end

# Generic case of no parameters
set_parameters(type::Type, position::Position) = type

"""
Set multiple type parameters.
"""
set_parameters(type::Type, parameters...) = set_parameters(type, Position(1), parameters...)
