# Base case, set the type parameter at the position if it is unspecified.
function set_unspecified_parameters(type::Type, position::Position, parameter)
  current_parameter = get_parameter(type, position)
  return replace_unspecified_parameters(type, position, current_parameter, parameter)
end

function replace_unspecified_parameters(
  type::Type, position::Position, current_parameter::Type{<:UnspecifiedParameter}, parameter
)
  return set_parameters(type, position, parameter)
end

function replace_unspecified_parameters(
  type::Type, position::Position, current_parameter, parameter
)
  return type
end

# Implementation in terms of generic version.
function set_unspecified_parameters(type::Type, position::Position, parameters...)
  return set_parameters(set_unspecified_parameters, type, position, parameters...)
end

"""
Set parameters starting from the first one if they are unspecified.
"""
function set_unspecified_parameters(type::Type, parameter...)
  return set_parameters(set_unspecified_parameters, type, Position(1), parameter...)
end
