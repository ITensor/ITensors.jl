"""
Represents the compile-time position of a type parameter.
"""
struct Position{x} end
Position(x) = Position{x}()

## `SetParameters` overloads for `Position`
get_parameter(::Type{<:Position{P}}, ::Position{1}) where {P} = P
set_parameter(::Type{<:Position}, ::Position{1}, P) = Position{P}
