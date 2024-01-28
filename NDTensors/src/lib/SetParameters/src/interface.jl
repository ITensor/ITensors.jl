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

"""
Required for removing all parameters.

Should return a `Type` with no parameters set
"""
function unspecify_parameters(type::Type)
  return error(
    "Unable to unspecify the paramters of type '$(type)' as it has not been defined."
  )
end

"""
Optional definitions for types which contain parameter eltype

Should return a `Type`.
"""
function set_eltype(type::Type)
  return error(
    "Unable to set the element type of type '$(type)' as it has not been defined."
  )
end

"""
Optional definitions for types which contain parameter ndim

Should return a `Type`.
"""
function set_ndims(type::Type)
  return error("Unable to set the ndim of type '$(type)' as it has not been defined.")
end

"""
Optional definitions for types which contain an `eltype` parameter. For the `set_eltype` function

  Should return a `Position`.
"""
function eltype_position(type::Type)
  return error(
    "Unable to find the parenttype position of type '$(type)' as it has not been defined."
  )
end

"""
Optional definitions for types which contain an `ndim` parameter. For the `set_ndims` function

  Should return a `Position`.
"""
function ndims_position(type::Type)
  return error(
    "Unable to find the parenttype position of type '$(type)' as it has not been defined."
  )
end

"""
Optional definitions for types which are considered `Wrappers` and have a `parenttype`

  Should return a `Position`.
"""
function parenttype_position(type::Type)
  return error(
    "Unable to find the parenttype position of type '$(type)' as it has not been defined."
  )
end
