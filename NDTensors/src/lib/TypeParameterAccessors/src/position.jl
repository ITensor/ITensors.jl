"""
Represents the compile-time position of a type parameter.
"""
struct Position{x} end
Position(x) = Position{x}()

"""
    parameter(type::Type, position::Position)

Get a type parameter of the type `type` at the position `position`.
"""
parameter(type::Type, position::Position) = parameter(type, Int(position))

set_parameter(type::Type, pos::Position, val) = set_parameter(type, parameter(pos), val)

Int(p::Position) = parameter(p)
