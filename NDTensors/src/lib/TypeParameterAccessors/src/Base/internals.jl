## Relies on Julia internals
@generated parameters(::Type{type}) where {type} =
  Tuple(Base.unwrap_unionall(type).parameters)
@generated unspecify_parameters(::Type{type}) where {type} = Base.typename(type).wrapper
# Type unstable version, for use within generated functions.
function _set_parameter(type::Type, pos, param)
  params = Base.setindex(parameters(type), param, pos)
  return Base.rewrap_unionall(unspecify_parameters(type){params...}, type)
end
# Convert a function type to an instance.
function_instance(ftype) = ftype.instance
