function set_parameters(type::DataType, parameters::Tuple)
  return unspecify_parameters(type){parameters...}
end
function set_parameters(type::UnionAll, parameters::Tuple)
  return datatype_to_unionall(set_parameters(unionall_to_datatype(type), parameters), type)
end

function set_parameter(type::Type, pos::Int, val)
  params = parameters(type)
  new_params = Base.setindex(params, val, pos)
  return set_parameters(type, new_params)
end

set_parameter(type::Type, pos::Position, val) = set_parameter(type, parameter(pos), val)

set_parameter(type::Type, val) = set_parameter(type, 1, val)
