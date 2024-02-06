"""
    parameters(object, position::Int)

Get a type parameter of the object `object` at the position `position`.
"""
parameters(type::DataType) = Tuple(type.parameters)

parameters(type::UnionAll) = parameters(unionall_to_datatype(type))

parameters(object) = parameters(typeof(object))

"""
    parameter(object, position::Int)

Get a type parameter of the object `object` at the position `position`.
"""
parameter(type, position::Int) = parameters(type)[position]

"""
    parameter(object, position::Position)

Get a type parameter of the object `object` at the position `position`.
"""
parameter(object::Type, position::Position) = parameter(object, parameter(position))

parameter(x) = parameter(typeof(x))

parameter(type::Type) = only(parameters(type))
 