"""
SmallVector
"""
struct SmallVector{S,T} <: AbstractSmallVector{T}
  buffer::SVector{S,T}
  length::Int
end

# Accessors
# TODO: Use `Accessors.jl`.
@inline setbuffer(vec::SmallVector, buffer) = SmallVector(buffer, vec.length)
@inline setlength(vec::SmallVector, length) = SmallVector(vec.buffer, length)

# Constructors
function SmallVector{S}(buffer::AbstractVector, len::Int) where {S}
  return SmallVector{S,eltype(buffer)}(buffer, len)
end
function SmallVector(buffer::AbstractVector, len::Int)
  return SmallVector{length(buffer),eltype(buffer)}(buffer, len)
end

"""
`SmallVector` constructor, uses `SVector` as buffer storage.
```julia
SmallVector{10}([1, 2, 3])
SmallVector{10}(SA[1, 2, 3])
```
"""
function SmallVector{S,T}(vec::AbstractVector) where {S,T}
  mvec = MSmallVector{S,T}(vec)
  return SmallVector{S,T}(buffer(mvec), length(mvec))
end
# Special optimization codepath for `MSmallVector`
# to avoid a copy.
function SmallVector{S,T}(vec::MSmallVector) where {S,T}
  return SmallVector{S,T}(buffer(vec), length(vec))
end

function SmallVector{S}(vec::AbstractVector) where {S}
  return SmallVector{S,eltype(vec)}(vec)
end

# Specialized constructor
function MSmallVector{S,T}(vec::SmallVector) where {S,T}
  return MSmallVector{S,T}(buffer(vec), length(vec))
end

# Derive the buffer length.
SmallVector(vec::AbstractSmallVector) = SmallVector{length(buffer(vec))}(vec)

Base.convert(::Type{T}, a::AbstractArray) where {T<:SmallVector} = a isa T ? a : T(a)::T

# Buffer interface
buffer(vec::SmallVector) = vec.buffer

# AbstractArray interface
Base.size(vec::SmallVector) = (vec.length,)

# Base overloads
@inline function Base.getindex(vec::SmallVector, index::Integer)
  @boundscheck checkbounds(vec, index)
  return @inbounds buffer(vec)[index]
end

Base.copy(vec::SmallVector) = vec

# Optimization, default uses `similar`.
Base.copymutable(vec::SmallVector) = MSmallVector(vec)
