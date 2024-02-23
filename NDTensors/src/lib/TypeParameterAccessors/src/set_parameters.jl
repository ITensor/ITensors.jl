# Generic function for modifying parameters at the specified positions,
# given a function for modifying a parameter at a position.
# This is type unstable and is meant to be used within a generated function.
function _set_parameters(f::Function, type::Type, positions::Tuple, params::Tuple)
  @assert length(positions) == length(params)
  for i in 1:length(positions)
    type = f(type, positions[i], params[i])
  end
  return type
end

## set_parameter
@generated function set_parameter(type::Type, pos::Position, param::TypeParameter)
  return _set_parameter(parameter(type), parameter(pos), parameter(param))
end
function set_parameter(type::Type, pos, param)
  return set_parameter(type, position(type, pos), TypeParameter(param))
end

function set_parameter(type::Type, param)
  return set_parameter(type, position(type, 1), TypeParameter(param))
end

## specify_parameter
function specify_parameter(type::Type, pos, val)
  is_parameter_specified(type, pos) && return type
  return set_parameter(type, pos, val)
end

# Because this is generated, this needs to convert function
# types to an instance of the function, which doesn't have a public API
# in Julia: https://discourse.julialang.org/t/is-there-a-way-to-get-f-from-typeof-f/18818
# In addition, a function passed as the first argument must be
# defined before this generated function is defined, or else
# there will be world age issues.
@generated function set_parameters(
  f::Function,
  type::Type,
  positions::Tuple{Vararg{Position}},
  params::Tuple{Vararg{TypeParameter}},
)
  return _set_parameters(
    function_instance(f),
    parameter(type),
    parameter.(parameters(positions)),
    parameter.(parameters(params)),
  )
end

function set_parameters(f::Function, type::Type, positions::Tuple, params::Tuple)
  return set_parameters(f, type, position.(type, positions), TypeParameter.(params))
end
function set_parameters(f::Function, type::Type, params::Tuple)
  return set_parameters(f, type, eachposition(type), params)
end

## set_parameters
function set_parameters(type::Type, positions::Tuple, params::Tuple)
  return set_parameters(set_parameter, type, positions, params)
end

set_parameters(type::Type, params::Tuple) = set_parameters(set_parameter, type, params)
set_parameters(type::Type) = type
