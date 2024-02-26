function specify_parameter(type::Type, pos, param)
  is_parameter_specified(type, pos) && return type
  return set_parameter(type, pos, param)
end

function _specify_parameters(type::Type, positions::Tuple{Vararg{Int}}, params::Tuple)
  new_params = parameters(type)
  for i in 1:length(positions)
    if !is_parameter_specified(type, positions[i])
      new_params = Base.setindex(new_params, params[i], positions[i])
    end
  end
  return new_parameters(type, new_params)
end
@generated function specify_parameters(
  type_type::Type,
  positions_type::Tuple{Vararg{Position}},
  params_type::Tuple{Vararg{TypeParameter}},
)
  type = parameter(type_type)
  positions = parameter.(parameters(positions_type))
  params = parameter.(parameters(params_type))
  return _specify_parameters(type, positions, params)
end
function specify_parameters(type::Type, positions::Tuple, params::Tuple)
  return specify_parameters(type, position.(type, positions), TypeParameter.(params))
end
function specify_parameters(type::Type, params::Tuple)
  return specify_parameters(type, eachposition(type), params)
end
