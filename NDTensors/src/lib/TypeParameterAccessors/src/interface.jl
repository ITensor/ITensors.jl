### These are optional interface functions which can be 
### defined on your type to make functions like `set_eltype` 
### usable

# Required overloads, generic fallback
position(::Type, ::Function) = UndefinedPosition()

function default_parameter(type::Type, fun::Function)
  return error("The default parameter of function $(fun) is not defined for type $(type)")
end

UnspecifiedFunction() = nothing

parameter_name(type::Type, p::Position) = error("There does not yet exist a name for the type $(type) at position $(int(p))")

parameter_name(type::Type, pos::Int) = parameter_name(type, Position(pos))
