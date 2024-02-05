unionall_to_datatype(type::Type) = Base.unwrap_unionall(type)
function datatype_to_unionall(type::Type, unionall_reference::Type)
  return Base.rewrap_unionall(type, unionall_reference)
end

unspecify_parameters(type::DataType) = Base.typename(type).wrapper

# `SetParameters` functionality.
datatype_to_unionall(type::Type) = datatype_to_unionall(type, unspecify_parameters(type))

function unspecify_parameters(type::UnionAll)
  return unspecify_parameters(unionall_to_datatype(type))
end

nparameters(type::Type) = length(parameters(type))

is_parameter_specified(type::Type, pos) = !(parameter(type, pos) isa TypeVar)

function set_parameters(type::DataType, parameters::Tuple)
  return unspecify_parameters(type){parameters...}
end
function set_parameters(type::UnionAll, parameters::Tuple)
  return datatype_to_unionall(set_parameters(unionall_to_datatype(type), parameters), type)
end

function set_parameter(type::Type, pos, val)
  params = parameters(type)
  new_params = Base.setindex(params, val, pos)
  return set_parameters(type, new_params)
end

function specify_parameter(type::Type, pos, val)
  is_parameter_specified(type, pos) && return type
  return set_parameter(type, pos, val)
end
