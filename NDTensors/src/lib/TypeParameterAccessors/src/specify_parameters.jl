Base.@assume_effects :foldable function specify_parameter(type::Type, pos::Int, param)
  return _specify_parameter(parameter(type, pos), type, Position(pos), param)
end

function _specify_parameter(::TypeVar, type::Type, pos::Position, param)
  return set_parameter(type, pos, TypeParameter(param))
end

function _specify_parameter(::TypeVar, type::Type, pos::Position, param::Type)
  return set_parameter(type, pos, param)
end

@generated function _specify_parameter(
  ::Union{<:Int,<:DataType}, ::Type{Typ}, pos, param
) where {Typ}
  return Typ
end

Base.@assume_effects :foldable function specify_parameter(type::Type, fun::Function, param)
  pos = position(type, fun)
  return _specify_parameter(parameter(type, pos), type, pos, param)
end

## TODO make a `specify_parameters(type::Type, functions, vals)`
# for (fun, val) in (functions, vals) specify_parameter(type, fun, val)
# function specify_parameters(type::Type, t...)
#   return type
# end
