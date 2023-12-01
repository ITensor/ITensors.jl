using Compat: Returns, allequal
using ..SparseArrayInterface: SparseArrayInterface

# `SparseArrayInterface` interface
SparseArrayInterface.sparse_storage(::AbstractDiagonalArray) = error("Not implemented")

# `AbstractArray` interface
Base.size(::AbstractDiagonalArray) = error("Not implemented")

function Base.similar(a::AbstractDiagonalArray, elt::Type, dims::Tuple{Vararg{Int}})
  return error("Not implemented")
end

function SparseArrayInterface.index_to_storage_index(
  a::AbstractDiagonalArray{<:Any,N}, I::CartesianIndex{N}
) where {N}
  !allequal(Tuple(I)) && return nothing
  return first(Tuple(I))
end

function SparseArrayInterface.storage_index_to_index(a::AbstractDiagonalArray, I)
  return CartesianIndex(ntuple(Returns(I), ndims(a)))
end

# 1-dimensional case can be `AbstractDiagonalArray`.
function SparseArrayInterface.sparse_similar(
  a::AbstractDiagonalArray, elt::Type, dims::Tuple{Int}
)
  return similar(a, elt, dims)
end

# AbstractArray interface
function Base.getindex(a::AbstractDiagonalArray, I...)
  return SparseArrayInterface.sparse_getindex(a, I...)
end

function Base.setindex!(a::AbstractDiagonalArray, I...)
  return SparseArrayInterface.sparse_setindex!(a, I...)
end

# AbstractArray functionality
# broadcast
function Broadcast.BroadcastStyle(arraytype::Type{<:AbstractDiagonalArray})
  return SparseArrayInterface.SparseArrayStyle{ndims(arraytype)}()
end

# map
function Base.map!(f, dest::AbstractArray, src::AbstractDiagonalArray)
  SparseArrayInterface.sparse_map!(f, dest, src)
  return dest
end

# permutedims
function Base.permutedims!(dest::AbstractArray, src::AbstractDiagonalArray, perm)
  SparseArrayInterface.sparse_permutedims!(dest, src, perm)
  return dest
end

# reshape
function Base.reshape(a::AbstractDiagonalArray, dims::Tuple{Vararg{Int}})
  return SparseArrayInterface.sparse_reshape(a, dims)
end
