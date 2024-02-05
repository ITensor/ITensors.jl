"""
    parameters(object, position::Int)

Get a type parameter of the object `object` at the position `position`.
"""
parameters(type::DataType) = Tuple(type.parameters)

parameters(type::UnionAll) = parameters(unionall_to_datatype(type))

"""
    get_parameter(object, position::Int)

Get a type parameter of the object `object` at the position `position`.
"""
get_parameter(type, position::Int) = parameters(type)[position]

"""
    get_parameter(object, position::Position)

Get a type parameter of the object `object` at the position `position`.
"""
get_parameter(object::Type, position::Position) =
  get_parameter(object, parameter(position))

"""
    get_parameter(type::Type)

Get the first type parameter of the type `type`.
"""
get_parameter(type::Type) = parameter(type, 1)

"""
    get_parameter(object)

Get the first type parameter of the object `object`.
"""
get_parameter(object) = get_parameter(typeof(object))

parameter(type::Type, pos::Int) = parameters(type)[pos]

parameter(type::Type, pos::Position) = parameters(type)[parameter(pos)]

parameter(x) = parameter(typeof(x))

parameter(type::Type) = only(parameters(type))
