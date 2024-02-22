### These are optional interface functions which can be 
### defined on your type to make functions like `set_eltype` 
### usable

# Required overloads, generic fallback
position(::Type, ::Function) = UndefinedPosition()

function default_parameter(type::Type, name)
  return error("The default parameter of $(name) is not defined for type $(type)")
end

UnspecifiedFunction() = nothing

function parameter_name(type::Type, p::Position)
  return error("There does not yet exist a name for the type $(type) at position $(int(p))")
end

parameter_name(type::Type, pos::Int) = parameter_name(type, Position(pos))
