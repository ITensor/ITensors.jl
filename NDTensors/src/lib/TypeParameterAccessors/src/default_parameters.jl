## default_parameters
function default_parameters(type::Type, positions::Tuple=eachposition(type))
  return map(pos -> default_parameter(type, pos), positions)
end

## set_default_parameter
function set_default_parameter(type::Type, pos)
  return set_parameter(type, pos, default_parameter(type, pos))
end

## set_default_parameters
function set_default_parameters(type::Type, positions::Tuple=eachposition(type))
  return set_parameters(type, positions, default_parameters(type, positions))
end

## specify_default_parameter
function specify_default_parameter(type::Type, pos)
  return specify_parameter(type, pos, default_parameter(type, pos))
end

## specify_default_parameters
function specify_default_parameters(type::Type, positions::Tuple=eachposition(type))
  return specify_parameters(type, positions, default_parameters(type, positions))
end