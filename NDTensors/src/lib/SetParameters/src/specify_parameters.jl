# Base case, set the type parameter at the position if it is unspecified.
function specify_parameters(type::Type, position::Position, parameter)
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
function specify_parameters(type::Type, position::Position, parameters...)
  return set_parameters(specify_parameters, type, position, parameters...)
end

"""
Set parameters starting from the first one if they are unspecified.
"""
function specify_parameters(type::Type, parameter...)
  return set_parameters(specify_parameters, type, Position(1), parameter...)
end

function specify_parameters(type::Type, parameters::Tuple)
  new_type = type
  for i in 1:(length(parameters) - 1)
    new_type = set_parameters(new_type, Position(i), parameters[i])
  end
  return new_type
end

function specify_parameters(type::Type)
  return specify_parameters(type, DefaultParameters())
end

function specify_parameter(type::Type, Position, parameter)
  return specify_parameters(type, Position, parameter)
end
