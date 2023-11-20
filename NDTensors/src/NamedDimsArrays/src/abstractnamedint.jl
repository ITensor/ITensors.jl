abstract type AbstractNamedInt{Value,Name} <: Integer end

# Interface
unname(i::AbstractNamedInt) = error("Not implemented")
name(i::AbstractNamedInt) = error("Not implemented")

# Derived
unname(::Type{<:AbstractNamedInt{Value}}) where {Value} = Value

# Integer interface
# TODO: Should this make a random name, or require defining a way
# to combine names?
Base.:*(i1::AbstractNamedInt, i2::AbstractNamedInt) = unname(i1) * unname(i2)
Base.:-(i::AbstractNamedInt) = typeof(i)(-unname(i), name(i))

# TODO: Define for `NamedInt`, `NamedUnitRange` fallback?
# Base.OneTo(stop::AbstractNamedInt) = namedoneto(stop)
## nameduniterange_type(::Type{<:AbstractNamedInt}) = error("Not implemented")

# TODO: Use conversion from `AbstractNamedInt` to `AbstractNamedUnitRange`
# instead of general `named`.
# Base.OneTo(stop::AbstractNamedInt) = namedoneto(stop)
Base.OneTo(stop::AbstractNamedInt) = named(Base.OneTo(unname(stop)), name(stop))

# TODO: Is this needed?
# Include the name as well?
Base.:<(i1::AbstractNamedInt, i2::AbstractNamedInt) = unname(i1) < unname(i2)
## Base.zero(type::Type{<:AbstractNamedInt}) = zero(unname(type))

function Base.promote_rule(type1::Type{<:AbstractNamedInt}, type2::Type{<:Integer})
  return promote_type(unname(type1), type2)
end
(type::Type{<:Integer})(i::AbstractNamedInt) = type(unname(i))
# TODO: Use conversion from `AbstractNamedInt` to `AbstractNamedUnitRange`
# instead of general `named`.
function Base.oftype(i1::AbstractNamedInt, i2::Integer)
  return named(convert(typeof(unname(i1)), i2), name(i1))
end

# Traits
isnamed(::AbstractNamedInt) = true
