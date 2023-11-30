using Compat: Returns, allequal
using ..SparseArrayInterface: SparseArrayInterface
# TODO: Put into `DiagonalArraysSparseArrayDOKsExt`?
using ..SparseArrayDOKs: SparseArrayDOK

# Minimal interface
SparseArrayInterface.storage(a::DiagonalArray) = a.diag

function SparseArrayInterface.index_to_storage_index(
  a::DiagonalArray{<:Any,N}, I::CartesianIndex{N}
) where {N}
  !allequal(Tuple(I)) && return nothing
  return first(Tuple(I))
end

function SparseArrayInterface.storage_index_to_index(a::DiagonalArray, I)
  return CartesianIndex(ntuple(Returns(I), ndims(a)))
end

# Defines similar when the output can't be `DiagonalArray`,
# such as in `reshape`.
# TODO: Put into `DiagonalArraysSparseArrayDOKsExt`?
# TODO: Special case 2D to output `SparseMatrixCSC`?
function SparseArrayInterface.sparse_similar(
  a::DiagonalArray, elt::Type, dims::Tuple{Vararg{Int}}
)
  return SparseArrayDOK{elt}(undef, dims)
end

# 1-dimensional case can be `DiagonalArray`.
function SparseArrayInterface.sparse_similar(a::DiagonalArray, elt::Type, dims::Tuple{Int})
  return similar(a, elt, dims)
end

# AbstractArray interface
Base.size(a::DiagonalArray) = a.dims

function Base.getindex(a::DiagonalArray, I...)
  return SparseArrayInterface.sparse_getindex(a, I...)
end

function Base.setindex!(a::DiagonalArray, I...)
  return SparseArrayInterface.sparse_setindex!(a, I...)
end

function Base.similar(a::DiagonalArray, elt::Type, dims::Tuple{Vararg{Int}})
  return DiagonalArray{elt}(undef, dims)
end

# AbstractArray functionality
# broadcast
function Broadcast.BroadcastStyle(arraytype::Type{<:DiagonalArray})
  return SparseArrayInterface.SparseArrayStyle{ndims(arraytype)}()
end

# map
function Base.map!(f, dest::AbstractArray, src::DiagonalArray)
  SparseArrayInterface.sparse_map!(f, dest, src)
  return dest
end

# permutedims
function Base.permutedims!(dest::AbstractArray, src::DiagonalArray, perm)
  SparseArrayInterface.sparse_permutedims!(dest, src, perm)
  return dest
end

# reshape
function Base.reshape(a::DiagonalArray, dims::Tuple{Vararg{Int}})
  return SparseArrayInterface.sparse_reshape(a, dims)
end
