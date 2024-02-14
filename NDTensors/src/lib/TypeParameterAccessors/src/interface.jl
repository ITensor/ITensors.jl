### These are optional interface functions which can be 
### defined on your type to make functions like `set_eltype` 
### usable

# Required overloads, generic fallback
position(::Type, ::Function) = UndefinedPosition()

function default_parameter(type::Type, fun::Function)
  return error("The default parameter of function $(fun) is not defined for type $(type)")
end

default_parameters(::Type)::Tuple = ()