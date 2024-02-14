Base.@assume_effects :foldable function specify_parameter(type::Type, pos::Int, param)
  return _specify_parameter(parameter(type, pos), type, Position(pos), param)
end

Base.@assume_effects :foldable function specify_parameter(type::Type, pos::Position, param)
  return _specify_parameter(parameter(type, pos), type, pos, param)
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

## Functions is a little confusing, You can actually
## put `Functions``, `Position``, or `Int`
## or a mixture of any like (eltype, Position(2), 3)
Base.@assume_effects :foldable function specify_parameters(type::Type, functions::Tuple, params::Tuple)
  @assert length(functions) == length(params)
  for l in 1:length(params)
    type = specify_parameter(type, functions[l], params[l])
  end
  return type
end

specify_defaults(type::Type) = specify_parameters(type, default_parameters(type))
