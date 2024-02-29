default_type_parameters(type::Type) = error("Not implemented")
function default_type_parameters(type::Type, positions::Tuple)
  return default_type_parameter.(type, positions)
end
default_type_parameters(object) = default_type_parameters(typeof(object))
function default_type_parameters(object, positions::Tuple)
  return default_type_parameters(typeof(object), positions)
end
function default_type_parameter(type::Type, pos::Position)
  return default_type_parameters(type)[Int(pos)]
end
function default_type_parameter(type::Type, pos)
  return default_type_parameter(type, position(type, pos))
end
function default_type_parameter(object, pos)
  return default_type_parameter(typeof(object), pos)
end

# Wrapping type parameters to improve type stability.
function wrapped_default_type_parameters(type::Type)
  return wrapped_type_parameter.(default_type_parameters(type))
end
function wrapped_default_type_parameters(type::Type, positions::Tuple)
  return wrapped_default_type_parameter.(type, positions)
end
wrapped_default_type_parameters(object) = wrapped_default_type_parameters(typeof(object))
function wrapped_default_type_parameters(object, positions::Tuple)
  return wrapped_default_type_parameters(typeof(object), positions)
end
function wrapped_default_type_parameter(type::Type, pos::Position)
  return wrapped_default_type_parameters(type)[Int(pos)]
end
function wrapped_default_type_parameter(type::Type, pos)
  return wrapped_default_type_parameter(type, position(type, pos))
end
function wrapped_default_type_parameter(object, pos)
  return wrapped_default_type_parameter(typeof(object), pos)
end

function set_default_type_parameter(type::Type, pos)
  return set_type_parameter(type, pos, wrapped_default_type_parameter(type, pos))
end
function set_default_type_parameters(type::Type)
  return set_type_parameters(type, wrapped_default_type_parameters(type))
end
function set_default_type_parameters(type::Type, positions::Tuple)
  return set_type_parameters(
    type, positions, wrapped_default_type_parameters(type, positions)
  )
end

function specify_default_type_parameter(type::Type, pos)
  return specify_type_parameter(type, pos, wrapped_default_type_parameter(type, pos))
end
function specify_default_type_parameters(type::Type)
  return specify_type_parameters(type, wrapped_default_type_parameters(type))
end
function specify_default_type_parameters(type::Type, positions::Tuple)
  return specify_type_parameters(
    type, positions, wrapped_default_type_parameters(type, positions)
  )
end
