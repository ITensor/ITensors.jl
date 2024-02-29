function _set_type_parameter(type::Type, pos::Int, param)
  params = Base.setindex(parameters(type), param, pos)
  return new_parameters(type, params)
end
@generated function set_type_parameter(
  type_type::Type, pos_type::Position, param_type::TypeParameter
)
  type = parameter(type_type)
  pos = parameter(pos_type)
  param = parameter(param_type)
  return _set_type_parameter(type, pos, param)
end
function set_type_parameter(type::Type, pos, param)
  return set_type_parameter(type, position(type, pos), param)
end
function set_type_parameter(type::Type, pos::Position, param)
  return set_type_parameter(type, pos, TypeParameter(param))
end
function set_type_parameter(type::Type, pos::Position, param::UnspecifiedTypeParameter)
  return unspecify_type_parameter(type, pos)
end

function _set_type_parameters(type::Type, positions::Tuple{Vararg{Int}}, params::Tuple)
  @assert length(positions) == length(params)
  new_params = parameters(type)
  for i in 1:length(positions)
    new_params = Base.setindex(new_params, params[i], positions[i])
  end
  return new_parameters(type, new_params)
end
@generated function set_type_parameters(
  type_type::Type,
  positions_type::Tuple{Vararg{Position}},
  params_type::Tuple{Vararg{TypeParameter}},
)
  type = parameter(type_type)
  positions = parameter.(parameters(positions_type))
  params = parameter.(parameters(params_type))
  return _set_type_parameters(type, positions, params)
end
function set_type_parameters(type::Type, positions::Tuple, params::Tuple)
  return set_type_parameters(type, position.(type, positions), TypeParameter.(params))
end
function set_type_parameters(type::Type, params::Tuple)
  return set_type_parameters(type, eachposition(type), params)
end
