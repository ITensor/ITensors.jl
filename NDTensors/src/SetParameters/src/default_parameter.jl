"""
Get the default parameter of an object at a specified position.
"""
default_parameter(object, position::Position) = default_parameter(typeof(object), position)

"""
Get the first default type parameter of a type.
"""
default_parameter(type::Type) = default_parameter(type, Position(1))

"""
Get the first default type parameter of an object.
"""
default_parameter(object) = default_parameter(typeof(object))

"""
Type to specify that some number of default parameters should be used.
Can use in place of specifying some number of parameters to set.

The type parameter `N` specifies the number of contiguous default parameters
to set. A value of `Any` means that the value will be inferred from the
type, using the `SetParameters.nparameters(type::Type)` function which
should be overloaded by new types.
"""
struct DefaultParameters{N}
  npositions::Val{N}
end
DefaultParameters() = DefaultParameters(Val(Any))
DefaultParameters(nparameters) = DefaultParameters(Val(nparameters))

(type::Type{<:DefaultParameters})() = DefaultParameters(get_parameter(type))

# `SetParameters` overload.
get_parameter(type::Type{<:DefaultParameters{P1}}, position::Position{1}) where {P1} = P1
function set_parameter(type::Type{<:DefaultParameters}, position::Position{1}, P1)
  return DefaultParameters{P1}
end

"""
`DefaultParameter` represents a single default parameter.
"""
const DefaultParameter = DefaultParameters{1}

function (parameter::DefaultParameter)(type::Type, position::Position)
  return default_parameter(type, position)
end

function Base.iterate(parameters::DefaultParameters, state=1)
  if state > get_parameter(parameters)
    return nothing
  end
  return (DefaultParameter(), state + 1)
end

function Base.iterate(parameters::DefaultParameters{Any}, state=1)
  return error("Can't iterate `$(parameters)`, must specify a number of parameters.")
end

# Specify the number of parameters.
# TODO: Check if this is type stable!
function set_nparameters(parameters::DefaultParameters{Any}, type::Type, position::Position)
  return set_nparameters(parameters, nparameters(type), position)
end

# Specify the number of parameters.
# TODO: Check if this is type stable!
function set_nparameters(
  parameters::DefaultParameters{Any}, nparameters::Val, position::Position
)
  ndefault_parameters = Val(get_parameter(nparameters) - get_parameter(position) + 1)
  return set_parameter(typeof(parameters), ndefault_parameters)()
end

function set_nparameters(parameters::DefaultParameters, type::Type, position::Position)
  return parameters
end

function set_nparameters(
  parameters::DefaultParameters, nparameters::Val, position::Position
)
  return parameters
end
