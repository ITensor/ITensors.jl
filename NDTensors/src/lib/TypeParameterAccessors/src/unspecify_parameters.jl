function _unspecify_parameters(type::Type)
  return Base.typename(type).wrapper
end
@generated function unspecify_parameters(type_type::Type)
  type = parameter(type_type)
  return _unspecify_parameters(type)
end

# Like `set_parameters` but less strict, i.e. it allows
# setting with `TypeVar` while `set_parameters` would error.
function new_parameters(type::Type, params)
  return to_unionall(unspecify_parameters(type){params...})
end

function _unspecify_parameter(type::Type, pos::Int)
  !is_parameter_specified(type, pos) && return type
  unspecified_param = parameter(unspecify_parameters(type), pos)
  params = Base.setindex(parameters(type), unspecified_param, pos)
  return new_parameters(type, params)
end
@generated function unspecify_parameter(type_type::Type, pos_type::Position)
  type = parameter(type_type)
  pos = parameter(pos_type)
  return _unspecify_parameter(type, pos)
end
function unspecify_parameter(type::Type, pos)
  return unspecify_parameter(type, position(type, pos))
end

function _unspecify_parameters(type::Type, positions::Tuple{Vararg{Int}})
  for pos in positions
    type = unspecify_parameter(type, pos)
  end
  return type
end
@generated function unspecify_parameters(
  type_type::Type, positions_type::Tuple{Vararg{Position}}
)
  type = parameter(type_type)
  positions = parameter.(parameters(positions_type))
  return _unspecify_parameters(type, positions)
end
function unspecify_parameters(type::Type, positions::Tuple)
  return unspecify_parameters(type, position.(type, positions))
end
