# Relies on Julia internals
Base.@assume_effects :terminates_globally to_datatype(type::UnionAll) =
  Base.unwrap_unionall(type)
Base.@assume_effects :terminates_globally to_unionall(
  type::DataType, unionall_reference::Type
) = Base.rewrap_unionall(type, unionall_reference)

# `TypeParameterAccessor` functionality.
to_datatype(type::DataType) = type
to_unionall(type::UnionAll, unionall_reference::Type) = type
to_unionall(type::UnionAll) = type
to_unionall(type::DataType) = to_unionall(type, unspecify_parameters(type))

# `TypeParameterAccessor` functionality.
to_unionall(type::Type) = to_unionall(type, unspecify_parameters(type))

unspecify_parameters(type::DataType) = Base.typename(type).wrapper

function unspecify_parameters(type::UnionAll)
  return unspecify_parameters(to_datatype(type))
end

is_parameter_specified(type::Type, pos) = !(parameter(type, pos) isa TypeVar)
