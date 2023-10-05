"""
A vector with a fixed maximum length, backed by a fixed size buffer.
"""
abstract type AbstractSmallVector{T} <: AbstractVector{T} end

# Required buffer interface
buffer(vec::AbstractSmallVector) = throw(NotImplemented())

similar_type(vec::AbstractSmallVector) = typeof(vec)

# Required buffer interface
maxlength(vec::AbstractSmallVector) = length(buffer(vec))
maxlength(vectype::Type{<:AbstractSmallVector}) = error("Not implemented")

function thaw_type(vectype::Type{<:AbstractSmallVector}, ::Type{T}) where {T}
  return MSmallVector{maxlength(vectype),T}
end
thaw_type(vectype::Type{<:AbstractSmallVector{T}}) where {T} = thaw_type(vectype, T)

# Required AbstractArray interface
Base.size(vec::AbstractSmallVector) = throw(NotImplemented())

# Derived AbstractArray interface
function Base.getindex(vec::AbstractSmallVector, index::Integer)
  return throw(NotImplemented())
end
function Base.setindex!(vec::AbstractSmallVector, item, index::Integer)
  return throw(NotImplemented())
end
Base.IndexStyle(::Type{<:AbstractSmallVector}) = IndexLinear()

function Base.convert(::Type{T}, a::AbstractArray) where {T<:AbstractSmallVector}
  return a isa T ? a : T(a)::T
end
