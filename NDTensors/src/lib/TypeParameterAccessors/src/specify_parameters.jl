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

Base.@assume_effects :foldable function specify_parameters(type::Type, functions::Tuple)
  for func in functions
    type = specify_parameter(type, func, default_parameter(type, func))
  end
  return type
end

# Base.@assume_effects :foldable function specify_parameters(type::Type, functions::Tuple, vals::Tuple)
#   @assert length(functions) == length(vals)
#   for l in 1:length(vals)
#     @show functions[l]
#     @show vals[l]
#     type = specify_parameter(type, functions[l], vals[l])
#   end
#   return type
# end

specify_defaults(type::Type) = specify_parameters(type, default_parameters(type))
