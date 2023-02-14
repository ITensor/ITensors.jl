"""
Set the specified parameter using the default parameters.
"""
function set_parameter(type::Type, position::Position, parameters::DefaultParameters)
  return set_parameter(type, position, default_parameter(type, position))
end

## """
## Set the first parameter using the default parameters.
## """
## set_parameter(type::Type, parameters::DefaultParameters) = set_parameter(type, Position(1), DefaultParameters())

function set_parameters(type::Type, start_position::Position, parameters::DefaultParameters)
  # Needed to get `generic_set_parameters` to loop over all
  # the possible parameters.
  unspecified_parameters = ntuple(Returns(nothing), nparameters(type))
  return generic_set_parameters(
    type, start_position, unspecified_parameters...
  ) do type, position, new_parameter
    return set_parameter(type, position, DefaultParameters())
  end
end
