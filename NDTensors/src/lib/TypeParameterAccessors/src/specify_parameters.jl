function specify_parameter(type::Type, pos::Int, param::Type)
  is_parameter_specified(type, pos) && return type
  return set_parameter(type, Position(pos), param)
end

function specify_parameter(type::Type, pos::Int, param)
  is_parameter_specified(type, pos) && return type
  return set_parameter(type, Position(pos), TypeParameter(param))
end

@generated specify_parameter(type::Type, pos::Position, param) =
  specify_parameter(type, Int(pos), param)

function specify_parameters(type::Type, t...)
  return type
end
