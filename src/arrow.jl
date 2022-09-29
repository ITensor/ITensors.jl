
"""
Arrow

`enum` type that can take three values: `In`, `Out`, or `Neither`, representing a directionality
associated with an index, i.e. the index leg is directed into or out of a given tensor
"""
@enum Arrow In = -1 Out = 1 Neither = 0

"""
    -(dir::Arrow)

Reverse direction of a directed `Arrow`.
"""
function Base.:-(dir::Arrow)
  return Arrow(-Int(dir))
end
