## Set an unspecified parameter to its default value
"""
Set to the default if it is unspecified.
"""
function set_unspecified_default_parameter(type::Type, position::Position)
  return set_unspecified_parameter(type, position, DefaultParameters())
end

"""
Set the first parameter to the default if it is unspecified.
"""
function set_unspecified_default_parameter(type::Type)
  return set_unspecified_parameter(type, Position(1), DefaultParameters())
end

# Set to the default if it is unspecified.
function set_unspecified_parameter(
  type::Type, position::Position, parameters::DefaultParameters
)
  return set_unspecified_parameter(type, position, default_parameter(type, position))
end

# Set the first parameter to the default if it is unspecified.
function set_unspecified_parameter(type::Type, parameters::DefaultParameters)
  return set_unspecified_parameter(type, Position(1), parameters)
end

## Set unspecified parameters to their default values
function set_unspecified_default_parameters(type::Type, start_position::Position)
  return set_unspecified_parameters(type, start_position, DefaultParameters())
end

function set_unspecified_default_parameters(type::Type)
  return set_unspecified_parameters(type, Position(1), DefaultParameters())
end

function set_unspecified_parameters(
  type::Type, start_position::Position, parameters::DefaultParameters
)
  # Needed to get `generic_set_parameters` to loop over all
  # the possible parameters.
  unspecified_parameters = ntuple(Returns(UnspecifiedParameter), nparameters(type))
  return generic_set_parameters(
    type, start_position, unspecified_parameters...
  ) do type, position, new_parameter
    return set_unspecified_parameter(type, position, DefaultParameters())
  end
end

function set_unspecified_parameters(type::Type, parameters::DefaultParameters)
  return set_unspecified_parameters(type, Position(1), parameters)
end
