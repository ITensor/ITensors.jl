### These are optional interface functions which can be 
### defined on your type to make functions like `set_eltype` 
### usable

# Required overloads, generic fallback
position(::Type, ::Function) = UndefinedPosition()
