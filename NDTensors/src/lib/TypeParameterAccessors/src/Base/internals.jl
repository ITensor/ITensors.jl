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

function to_unionall(type::Type)
  params = Base.unwrap_unionall(type).parameters
  for i in reverse(eachindex(params))
    param = params[i]
    if param isa TypeVar
      type = UnionAll(param, type)
    end
  end
  return type
end

function unspecify_parameter(type::Type, pos::Int)
  params = Base.unwrap_unionall(type).parameters
  if !(params[pos] isa TypeVar)
    unspecified_param = Base.unwrap_unionall(Base.typename(type).wrapper).parameters[pos]
    return to_unionall(
      Base.typename(type).wrapper{
        params[1:(pos - 1)]...,unspecified_param,params[(pos + 1):end]...
      },
    )
  end
  return type
end
