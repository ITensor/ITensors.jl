"""
    parameters(type::DataType)

Gets all the type parameters of the DataType `type`.
"""
parameters(type::DataType) = Tuple(type.parameters)

"""
    parameters(type::UnionAll, position::Int)

Gets all the type parameters of the UnionAll `type`.
"""
parameters(type::UnionAll) = parameters(unionall_to_datatype(type))

"""
    parameters(object, position::Int)

Gets all the type parameters of the object `object`.
"""
parameters(object) = parameters(typeof(object))

"""
    parameter(type::Type, position::Int)

Get a type parameter of the type `type` at the position `position`.
"""
parameter(type::Type, position::Int) = parameters(type)[position]

"""
    parameter(type::Type, position::Position)

Get a type parameter of the type `type` at the position `position`.
"""
parameter(type::Type, position::Position) = parameter(object, parameter(position))

"""
    parameter(type::Type)

Gets the single parameter of the Type `type`. Will throw an error if `type` has more than one parameter.
"""
parameter(type::Type) = only(parameters(type))

"""
    parameter(type::Type)

Gets the single parameter of the object `object`. Will throw an error if `object` has more than one parameter.
"""
parameter(x) = parameter(typeof(x))
 