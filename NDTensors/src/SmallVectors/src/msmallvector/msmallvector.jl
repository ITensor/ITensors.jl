"""
MSmallVector
"""
mutable struct MSmallVector{S,T} <: AbstractSmallVector{T}
  const buffer::MVector{S,T}
  length::Int
end

# Constructors
function MSmallVector{S}(buffer::AbstractVector, len::Int) where {S}
  return MSmallVector{S,eltype(buffer)}(buffer, len)
end
function MSmallVector(buffer::AbstractVector, len::Int)
  return MSmallVector{length(buffer),eltype(buffer)}(buffer, len)
end

"""
`MSmallVector` constructor, uses `MVector` as a buffer.
```julia
MSmallVector{10}([1, 2, 3])
MSmallVector{10}(SA[1, 2, 3])
```
"""
function MSmallVector{S,T}(vec::AbstractVector) where {S,T}
  buffer = zeros(MVector{S,T})
  copyto!(buffer, vec)
  return MSmallVector(buffer, length(vec))
end

# Derive the buffer length.
MSmallVector(vec::AbstractSmallVector) = MSmallVector{length(buffer(vec))}(vec)

Base.convert(::Type{T}, a::AbstractArray) where {T<:MSmallVector} = a isa T ? a : T(a)::T

function MSmallVector{S}(vec::AbstractVector) where {S}
  return MSmallVector{S,eltype(vec)}(vec)
end

function MSmallVector{S,T}(::UndefInitializer, dims::Tuple{Integer}) where {S,T}
  return MSmallVector{S,T}(undef, prod(dims))
end
function MSmallVector{S,T}(::UndefInitializer, length::Integer) where {S,T}
  return MSmallVector{S,T}(MVector{S,T}(undef), length)
end

# Buffer interface
buffer(vec::MSmallVector) = vec.buffer

# Accessors
Base.size(vec::MSmallVector) = (vec.length,)

# Required Base overloads
@inline function Base.getindex(vec::MSmallVector, index::Integer)
  @boundscheck checkbounds(vec, index)
  return @inbounds buffer(vec)[index]
end

@inline function Base.setindex!(vec::MSmallVector, item, index::Integer)
  @boundscheck checkbounds(vec, index)
  @inbounds buffer(vec)[index] = item
  return vec
end

@inline function Base.resize!(vec::MSmallVector, len::Integer)
  len < 0 && throw(ArgumentError("New length must be ≥ 0."))
  len > maxlength(vec) &&
    throw(ArgumentError("New length $len must be ≤ the maximum length $(maxlength(vec))."))
  vec.length = len
  return vec
end

# `similar` creates a `MSmallVector` by default.
function Base.similar(vec::AbstractSmallVector, elt::Type, dims::Dims)
  return MSmallVector{length(buffer(vec)),elt}(undef, dims)
end
