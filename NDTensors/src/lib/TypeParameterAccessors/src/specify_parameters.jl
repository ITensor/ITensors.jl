## specify_parameters
function specify_parameters(type::Type, positions::Tuple, params::Tuple)
  return set_parameters(specify_parameter, type, positions, params)
end
function specify_parameters(type::Type, params::Tuple)
  return set_parameters(specify_parameter, type, params)
end