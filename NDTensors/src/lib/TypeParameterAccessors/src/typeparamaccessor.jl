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

function specify_parameter(type::Type, pos, val)
  is_parameter_specified(type, pos) && return type
  return set_parameter(type, pos, val)
end
