function default_type_parameter(type::Type, pos::Position)
  return default_type_parameter(type, position_name(type, pos))
end
default_type_parameter(type::Type, pos) = default_type_parameter(type, position(type, pos))
function default_type_parameter(type::Type, pos::UndefinedPosition)
  return UnspecifiedTypeParameter()
end
default_type_parameter(object, pos) = default_type_parameter(typeof(object), pos)

function default_type_parameters(type_or_object, positions::Tuple{Vararg{Position}})
  return map(pos -> default_type_parameter(type_or_object, pos), positions)
end
function default_type_parameters(
  type_or_object, positions::Tuple=eachposition(type_or_object)
)
  return default_type_parameters(type_or_object, position.(type_or_object, positions))
end

function set_default_parameter(type::Type, pos)
  return set_parameter(type, pos, default_type_parameter(type, pos))
end

function set_default_parameters(type::Type, positions::Tuple=eachposition(type))
  return set_parameters(type, positions, default_type_parameters(type, positions))
end

function specify_default_parameter(type::Type, pos)
  return specify_parameter(type, pos, default_type_parameter(type, pos))
end

function specify_default_parameters(type::Type, positions::Tuple=eachposition(type))
  return specify_parameters(type, positions, default_type_parameters(type, positions))
end
