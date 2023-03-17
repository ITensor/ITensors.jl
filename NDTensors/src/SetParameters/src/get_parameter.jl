"""
    get_parameter(object, position::Position)

Get a type parameter of the object `object` at the position `position`.
"""
get_parameter(object, position::Position) = get_parameter(typeof(object), position)

"""
    get_parameter(type::Type)

Get the first type parameter of the type `type`.
"""
get_parameter(type::Type) = get_parameter(type, Position(1))

"""
    get_parameter(object)

Get the first type parameter of the object `object`.
"""
get_parameter(object) = get_parameter(typeof(object))

# TODO: Define `get_parameters`?
# function get_parameters(type::Type)
#   return ntuple(nparameters(type)) do position
#     get_parameter(type, Position(position))
#   end
# end
