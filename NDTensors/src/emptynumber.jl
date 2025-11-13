#
# Represents a number that can be set to any type.
#

struct EmptyNumber <: Real end

zero(::Type{EmptyNumber}) = EmptyNumber()
zero(n::EmptyNumber) = zero(typeof(n))

# This helps handle a lot of basic algebra, like:
# EmptyNumber() + 2.3 == 2.3
convert(::Type{T}, x::EmptyNumber) where {T <: Number} = T(zero(T))

# TODO: Should this be implemented?
#Complex(x::Real, ::EmptyNumber) = x

# This is to help define `float(::EmptyNumber) = 0.0`.
# This helps with defining `norm` of `EmptyStorage{EmptyNumber}`.
AbstractFloat(::EmptyNumber) = zero(AbstractFloat)

# Extra definitions fix ambiguity errors.
Base.promote_rule(::Type{EmptyNumber}, T::Type{<:Number}) = T
Base.promote_rule(T::Type{<:Number}, ::Type{EmptyNumber}) = T
Base.promote_rule(::Type{EmptyNumber}, ::Type{Bool}) = Bool
Base.promote_rule(::Type{Bool}, ::Type{EmptyNumber}) = Bool
Base.promote_rule(::Type{EmptyNumber}, T::Type{Complex{R}}) where {R <: Real} = T
Base.promote_rule(T::Type{Complex{R}}, ::Type{EmptyNumber}) where {R <: Real} = T

# Basic arithmetic
(::EmptyNumber + ::EmptyNumber) = EmptyNumber()
(::EmptyNumber - ::EmptyNumber) = EmptyNumber()
(::Number * ::EmptyNumber) = EmptyNumber()
(::EmptyNumber * ::Number) = EmptyNumber()
(::EmptyNumber * ::EmptyNumber) = EmptyNumber()
(::EmptyNumber / ::Number) = EmptyNumber()
(::Number / ::EmptyNumber) = throw(DivideError())
(::EmptyNumber / ::EmptyNumber) = throw(DivideError())
-(::EmptyNumber) = EmptyNumber()
