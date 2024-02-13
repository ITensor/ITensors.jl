"""
Represents an position for type parameter which has not been registered.
"""
struct UndefinedPosition end

# @generated set_parameter(type::Type{Typ}, ::UndefinedPosition, val) where {Typ} = type
