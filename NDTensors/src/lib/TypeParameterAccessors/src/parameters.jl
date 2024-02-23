## TypeParmater
struct TypeParameter{Param} end
TypeParameter(param) = TypeParameter{param}()

## parameters
parameter(type::Type) = only(parameters(type))

## nparameters
nparameters(type::Type) = length(parameters(type))

parameters(object) = parameters(typeof(object))
## parameter
parameter(object) = parameter(typeof(object))
# Named position.
parameter(type::Type, pos) = parameter(type, position(type, pos))
parameter(type::Type, pos::Position) = parameters(type)[Int(pos)]
parameter(type::Type, pos::Int) = parameter(type, Position(pos))

## is_parameter_specified
is_specified_parameter(param) = true
is_specified_parameter(param::TypeVar) = false
is_parameter_specified(type::Type, pos) = is_specified_parameter(parameter(type, pos))

## get_parameter
function get_parameter(type::Type, pos)
  param = parameter(type, pos)
  !is_specified_parameter(param) && error("The requested type parameter isn't specified.")
  return param
end
