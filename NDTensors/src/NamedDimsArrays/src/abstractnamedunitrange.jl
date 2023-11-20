abstract type AbstractNamedUnitRange{T,Value<:AbstractUnitRange{T},Name} <:
              AbstractUnitRange{T} end

# Required interface
unname(::AbstractNamedUnitRange) = error("Not implemented")
name(::AbstractNamedUnitRange) = error("Not implemented")

# Traits
isnamed(::AbstractNamedUnitRange) = true

# Unit range
Base.first(i::AbstractNamedUnitRange) = first(unname(i))
Base.last(i::AbstractNamedUnitRange) = last(unname(i))
Base.length(i::AbstractNamedUnitRange) = named(length(unname(i)), name(i))
