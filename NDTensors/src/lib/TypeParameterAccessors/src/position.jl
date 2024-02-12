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

"""
    set_parameter(::Type{Typ}, ::Position{Pos}, ::Type{Param})

Sets the parameter at Position `Pos` of the Type `Typ` to the new Type `param`.
This function is necessary to ensure type stability.
"""
@generated function set_parameter(
  ::Type{Typ}, ::Position{Pos}, ::Type{Param}
) where {Typ,Pos,Param}
  return set_parameter(Typ, Pos, Param)
end

struct TypeParameter{P} end
TypeParameter(x) = TypeParameter{x}()

"""
    set_parameter(::Type{Typ}, ::Position{Pos}, ::TypeParameter{Param})

Sets the parameter at Position `Pos` of the Type `Typ` to the new Type `param`.
This function is necessary to ensure type stability.
"""
@generated function set_parameter(
  ::Type{Typ}, ::Position{Pos}, ::TypeParameter{Param}
) where {Typ,Pos,Param}
  return set_parameter(Typ, Pos, Param)
end

int(p::Position) = parameter(p)

@generated specify_parameter(type::Type, pos::Position, param) =
  specify_parameter(type, Int(pos), param)
