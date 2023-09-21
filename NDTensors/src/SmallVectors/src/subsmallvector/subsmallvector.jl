abstract type AbstractSubSmallVector{T} <: AbstractSmallVector{T} end

"""
SubSmallVector
"""
struct SubSmallVector{T,P} <: AbstractSubSmallVector{T}
  parent::P
  start::Int
  stop::Int
end

mutable struct SubMSmallVector{T,P<:AbstractVector{T}} <: AbstractSubSmallVector{T}
  const parent::P
  start::Int
  stop::Int
end

# TODO: Use Accessors.jl
Base.parent(vec::SubSmallVector) = vec.parent
Base.parent(vec::SubMSmallVector) = vec.parent

# buffer interface
buffer(vec::AbstractSubSmallVector) = buffer(parent(vec))

smallview(vec::SmallVector, start::Integer, stop::Integer) = SubSmallVector(vec, start, stop)
smallview(vec::MSmallVector, start::Integer, stop::Integer) = SubMSmallVector(vec, start, stop)

function smallview(vec::SubSmallVector, start::Integer, stop::Integer)
  return SubSmallVector(parent(vec), vec.start + start - 1, vec.start + stop - 1)
end
function smallview(vec::SubMSmallVector, start::Integer, stop::Integer)
  return SubMSmallVector(parent(vec), vec.start + start - 1, vec.start + stop - 1)
end

# Constructors
SubSmallVector(vec::AbstractVector, start::Integer, stop::Integer) = SubSmallVector{eltype(vec),typeof(vec)}(vec, start, stop)
SubMSmallVector(vec::AbstractVector, start::Integer, stop::Integer) = SubMSmallVector{eltype(vec),typeof(vec)}(vec, start, stop)

# Accessors
Base.size(vec::AbstractSubSmallVector) = (vec.stop - vec.start + 1,)

Base.@propagate_inbounds function Base.getindex(vec::AbstractSubSmallVector, index::Integer)
  return parent(vec)[index + vec.start - 1]
end

Base.@propagate_inbounds function Base.setindex!(vec::AbstractSubSmallVector, item, index::Integer)
  buffer(vec)[index + vec.start - 1] = item
  return vec
end

function SubSmallVector{T,P}(vec::SubMSmallVector) where {T,P}
  return SubSmallVector{T,P}(P(parent(vec)), vec.start, vec.stop)
end

function Base.convert(smalltype::Type{<:SubSmallVector}, vec::SubMSmallVector)
  return smalltype(vec)
end

@inline function Base.resize!(vec::SubMSmallVector, len::Integer)
  len < 0 && throw(ArgumentError("New length must be ≥ 0."))
  len > maxlength(vec) - vec.start + 1 && throw(ArgumentError("New length $len must be ≤ the maximum length $(maxlength(vec))."))
  vec.stop = vec.start + len - 1
  return vec
end

# Optimization, default uses `similar`.
function Base.copymutable(vec::SubSmallVector)
  return SubMSmallVector(Base.copymutable(parent(vec)), vec.start, vec.stop)
end
