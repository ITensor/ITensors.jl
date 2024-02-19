"""
Represents the compile-time position of a type parameter.
"""
struct Position{x} end
Position(x) = Position{x}()

"""
    parameter(type::Type, position::Position)

Get a type parameter of the type `type` at the position `position`.
"""
parameter(type::Type, position::Position) = parameter(type, int(position))

struct TypeParameter{P} end
TypeParameter(x) = TypeParameter{x}()

int(p::Position) = parameter(p)

eachposition(type::Type) = ntuple(Position, nparameters(type))

parameter_name(type::Type) = Base.Fix1(parameter_name, type)
parameter_names(type::Type) = map(parameter_name(type), eachposition(type))
