function _set_parameter(type::Type, pos::Int, param)
  params = Base.setindex(parameters(type), param, pos)
  return new_parameters(type, params)
end
@generated function set_parameter(
  type_type::Type, pos_type::Position, param_type::TypeParameter
)
  type = parameter(type_type)
  pos = parameter(pos_type)
  param = parameter(param_type)
  return _set_parameter(type, pos, param)
end
function set_parameter(type::Type, pos, param)
  return set_parameter(type, position(type, pos), param)
end
function set_parameter(type::Type, pos::Position, param)
  return set_parameter(type, pos, TypeParameter(param))
end
function set_parameter(type::Type, pos::Position, param::UnspecifiedTypeParameter)
  return unspecify_parameter(type, pos)
end
