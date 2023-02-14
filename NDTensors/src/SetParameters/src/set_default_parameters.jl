## Setting default type parameters.
# Base case, only specify one default parameter. Actually extracts the defined default
# parameter for the given type.
# `DefaultParameter = DefaultParameters{1}`
function set_parameters(type::Type, position::Position, parameter::DefaultParameter)
  return set_parameters(type, position, parameter(type, position))
end

# Catch cases when number of default parameters aren't specified,
# and determine them from the type and the position.
function set_parameters(type::Type, position::Position, parameters::DefaultParameters{Any})
  return set_parameters(set_parameters, type, position, parameters)
end

"""
Set multiple default type parameters.
"""
function set_parameters(type::Type, position::Position, parameters::DefaultParameters)
  return set_parameters(set_parameters, type, position, parameters)
end

# Set multiple default type parameters of any number.
# Set automatically to `nparameters`.
function set_parameters(
  set_parameter_function::Function,
  type::Type,
  position::Position,
  parameters::DefaultParameters{Any},
)
  return set_parameters(
    set_parameter_function, type, position, set_nparameters(parameters, type, position)
  )
end

# Set parameters starting from position `position`.
function set_parameters(
  set_parameter_function::Function,
  type::Type,
  position::Position,
  parameters::DefaultParameters,
)
  return set_parameters(set_parameter_function, type, position, parameters...)
end

# Base case.
function set_parameters(
  set_parameter_function::Function,
  type::Type,
  position::Position,
  parameter::DefaultParameter,
)
  return set_parameter_function(type, position, parameter)
end
