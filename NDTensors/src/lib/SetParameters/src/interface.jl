"""
Required interface for getting parameters.
"""
get_parameter(type::Type, position::Position) = UnspecifiedParameter

"""
Required interface for setting parameters.
"""
function set_parameter(type::Type, position::Position, parameter)
  return error(
    "Setting the type parameter of the type `$(type)` at position `$(position)` to `$(parameter)` is not currently defined. Either that type parameter position doesn't exist in the type, or `set_parameter` has not been overloaded for this type.",
  )
end

"""
Required for setting one or more default parameters.
"""
function default_parameter(type::Type, position::Position)
  return error(
    "The default type parameter of the type `$(type)` at position `$(position)` has not been defined.",
  )
end

"""
Required for setting multiple default parameters.

Should return a `Val`.
"""
function nparameters(type::Type)
  return error("The number of type parameters of the type `$(type)` has not been defined.")
end
