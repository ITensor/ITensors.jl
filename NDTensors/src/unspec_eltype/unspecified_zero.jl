#
# Represents a number that can be set to any type.
#

struct UnspecifiedZero <: Base.AbstractFloat end

zero(::Type{UnspecifiedZero}) = UnspecifiedZero()
zero(n::UnspecifiedZero) = zero(typeof(n))

# This helps handle a lot of basic algebra, like:
# UnspecifiedZero() + 2.3 == 2.3
convert(::Type{T}, x::UnspecifiedZero) where {T<:Number} = T(zero(T))

# TODO: Should this be implemented?
#Complex(x::Real, ::UnspecifiedZero) = x

# This is to help define `float(::UnspecifiedZero) = 0.0`.
# This helps with defining `norm` of `EmptyStorage{UnspecifiedZero}`.
AbstractFloat(::UnspecifiedZero) = zero(AbstractFloat)

# Basic arithmetic
(::UnspecifiedZero + ::UnspecifiedZero) = UnspecifiedZero()
(::UnspecifiedZero - ::UnspecifiedZero) = UnspecifiedZero()
(::Number * ::UnspecifiedZero) = UnspecifiedZero()
(::UnspecifiedZero * ::Number) = UnspecifiedZero()
(::UnspecifiedZero * ::UnspecifiedZero) = UnspecifiedZero()
(::UnspecifiedZero / ::Number) = UnspecifiedZero()
(::Number / ::UnspecifiedZero) = throw(DivideError())
(::UnspecifiedZero / ::UnspecifiedZero) = throw(DivideError())
-(::UnspecifiedZero) = UnspecifiedZero()

Base.promote_type(::Type{<:UnspecifiedZero}, ElT::Type) = ElT
Base.promote_type(ElT::Type, ::Type{<:UnspecifiedZero}) = ElT

Base.promote_type(::Type{<:UnspecifiedZero}, ::Type{<:UnspecifiedZero}) = UnspecifiedZero
