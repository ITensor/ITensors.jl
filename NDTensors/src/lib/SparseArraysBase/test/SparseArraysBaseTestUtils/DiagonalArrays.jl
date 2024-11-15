module DiagonalArrays
using NDTensors.SparseArraysBase: SparseArraysBase

struct DiagonalArray{T,N} <: AbstractArray{T,N}
  data::Vector{T}
  dims::Tuple{Vararg{Int,N}}
end
function DiagonalArray{T,N}(::UndefInitializer, dims::Tuple{Vararg{Int,N}}) where {T,N}
  return DiagonalArray{T,N}(Vector{T}(undef, minimum(dims)), dims)
end
function DiagonalArray{T,N}(::UndefInitializer, dims::Vararg{Int,N}) where {T,N}
  return DiagonalArray{T,N}(undef, dims)
end
function DiagonalArray{T}(::UndefInitializer, dims::Tuple{Vararg{Int}}) where {T}
  return DiagonalArray{T,length(dims)}(undef, dims)
end
function DiagonalArray{T}(::UndefInitializer, dims::Vararg{Int}) where {T}
  return DiagonalArray{T}(undef, dims)
end

# AbstractArray interface
Base.size(a::DiagonalArray) = a.dims
function Base.getindex(a::DiagonalArray, I...)
  return SparseArraysBase.sparse_getindex(a, I...)
end
function Base.setindex!(a::DiagonalArray, I...)
  return SparseArraysBase.sparse_setindex!(a, I...)
end
function Base.similar(a::DiagonalArray, elt::Type, dims::Tuple{Vararg{Int}})
  return DiagonalArray{elt}(undef, dims)
end

# Minimal interface
SparseArraysBase.sparse_storage(a::DiagonalArray) = a.data
function SparseArraysBase.index_to_storage_index(
  a::DiagonalArray{<:Any,N}, I::CartesianIndex{N}
) where {N}
  !allequal(Tuple(I)) && return nothing
  return first(Tuple(I))
end
function SparseArraysBase.storage_index_to_index(a::DiagonalArray, I)
  return CartesianIndex(ntuple(Returns(I), ndims(a)))
end
function SparseArraysBase.sparse_similar(
  a::DiagonalArray, elt::Type, dims::Tuple{Vararg{Int}}
)
  return Array{elt}(undef, dims)
end
function SparseArraysBase.sparse_similar(a::DiagonalArray, elt::Type, dims::Tuple{Int})
  return similar(a, elt, dims)
end

# Broadcasting
function Broadcast.BroadcastStyle(arraytype::Type{<:DiagonalArray})
  return SparseArraysBase.SparseArrayStyle{ndims(arraytype)}()
end

# Base
function Base.iszero(a::DiagonalArray)
  return SparseArraysBase.sparse_iszero(a)
end
function Base.isreal(a::DiagonalArray)
  return SparseArraysBase.sparse_isreal(a)
end
function Base.zero(a::DiagonalArray)
  return SparseArraysBase.sparse_zero(a)
end
function Base.one(a::DiagonalArray)
  return SparseArraysBase.sparse_one(a)
end
function Base.:(==)(a1::DiagonalArray, a2::DiagonalArray)
  return SparseArraysBase.sparse_isequal(a1, a2)
end
function Base.reshape(a::DiagonalArray, dims::Tuple{Vararg{Int}})
  return SparseArraysBase.sparse_reshape(a, dims)
end

# Map
function Base.map!(f, dest::AbstractArray, src::DiagonalArray)
  SparseArraysBase.sparse_map!(f, dest, src)
  return dest
end
function Base.copy!(dest::AbstractArray, src::DiagonalArray)
  SparseArraysBase.sparse_copy!(dest, src)
  return dest
end
function Base.copyto!(dest::AbstractArray, src::DiagonalArray)
  SparseArraysBase.sparse_copyto!(dest, src)
  return dest
end
function Base.permutedims!(dest::AbstractArray, src::DiagonalArray, perm)
  SparseArraysBase.sparse_permutedims!(dest, src, perm)
  return dest
end
end
