"""
    parameters(type::UnionAll, position::Int)

Gets all the type parameters of the UnionAll `type`.
"""
parameters(type::UnionAll) = parameters(to_datatype(type))

"""
    parameters(object, position::Int)

Gets all the type parameters of the object `object`.
"""
parameters(object) = parameters(typeof(object))

struct SpecifiedParameter{N} end
SpecifiedParameter(val) = SpecifiedParameter{is_specified_parameter(val)}()

is_specified_parameter(param)::Bool = true
is_specified_parameter(param::TypeVar)::Bool = false
is_parameter_specified(type::Type, pos) = is_specified_parameter(parameter(type, pos))

"""
    parameter(type::Type, position::Int)

Get a type parameter of the type `type` at the position `position`.
"""
parameter(type::Type, position::Int) = parameters(type)[position]

"""
    parameter(type::Type, position::Int)

Get a type parameter of the type `type` at the position `position`.
"""
parameter(type::Type, func::Function) = parameter(type, position(type, func))

function parameter(type::Type, position::UndefinedPosition)
  return error(
    "Unable to recover the parameter of an UndefinedPosition. If you are trying to access a position through a function, please register the function `position(::Type{<:$(type)}, ::Function)`.",
  )
end

"""
    parameter(type::Type)

Gets the single parameter of the Type `type`. Will throw an error if `type` has more than one parameter.
"""
parameter(type::Type) = only(parameters(type))

"""
    parameter(type::Type)

Gets the single parameter of the object `object`. Will throw an error if `object` has more than one parameter.
"""
parameter(x) = parameter(typeof(x))

"""
    nparameter(type::Type)

Gets the number of parameters for the Type `type`.
"""
nparameters(type::Type) = length(parameters(type))

function get_parameter(type::Type, pos)
  param = parameter(type, pos)
  return _get_parameter(SpecifiedParameter(param), param)
end

function _get_parameter(::SpecifiedParameter{true}, parameter)
  return parameter
end

function _get_parameter(::SpecifiedParameter{false}, parameter)
  return error("The requested type parameter isn't specified.")
end
