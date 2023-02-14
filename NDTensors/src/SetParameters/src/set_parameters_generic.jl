# Generic `set_parameters` that sets the parameters one by one
# using a custom parameter set function.
# `set_parameter` should have the signature:
# ```julia
# set_parameter(type::Type, position::Position, parameter)
# ```
function set_parameters(set_parameter_function::Function, type::Type, parameters...)
  return set_parameters(set_parameter_function, type, Position(1), parameters...)
end

# Set parameters starting from position `position`.
function set_parameters(
  set_parameter_function::Function,
  type::Type,
  position::Position,
  parameter1,
  parameters_tail...,
)
  new_type = set_parameter_function(type, position, parameter1)
  new_position = Position(get_parameter(position) + 1)
  return set_parameters(set_parameter_function, new_type, new_position, parameters_tail...)
end

# Stop recursion, no more parameters to set.
set_parameters(set_parameter_function::Function, type::Type, position::Position) = type
