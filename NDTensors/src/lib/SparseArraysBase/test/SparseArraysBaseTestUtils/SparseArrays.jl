module SparseArrays
using LinearAlgebra: LinearAlgebra
using NDTensors.SparseArraysBase: SparseArraysBase, Zero

struct SparseArray{T,N,Zero} <: AbstractArray{T,N}
  data::Vector{T}
  dims::Tuple{Vararg{Int,N}}
  index_to_dataindex::Dict{CartesianIndex{N},Int}
  dataindex_to_index::Vector{CartesianIndex{N}}
  zero::Zero
end
function SparseArray{T,N}(dims::Tuple{Vararg{Int,N}}; zero=Zero()) where {T,N}
  return SparseArray{T,N,typeof(zero)}(
    T[], dims, Dict{CartesianIndex{N},Int}(), Vector{CartesianIndex{N}}(), zero
  )
end
function SparseArray{T,N}(dims::Vararg{Int,N}; kwargs...) where {T,N}
  return SparseArray{T,N}(dims; kwargs...)
end
function SparseArray{T}(dims::Tuple{Vararg{Int}}; kwargs...) where {T}
  return SparseArray{T,length(dims)}(dims; kwargs...)
end
function SparseArray{T}(::UndefInitializer, dims::Tuple{Vararg{Int}}; kwargs...) where {T}
  return SparseArray{T}(dims; kwargs...)
end
SparseArray{T}(dims::Vararg{Int}; kwargs...) where {T} = SparseArray{T}(dims; kwargs...)

# LinearAlgebra interface
function LinearAlgebra.mul!(
  a_dest::AbstractMatrix,
  a1::SparseArray{<:Any,2},
  a2::SparseArray{<:Any,2},
  α::Number,
  β::Number,
)
  SparseArraysBase.sparse_mul!(a_dest, a1, a2, α, β)
  return a_dest
end

function LinearAlgebra.dot(a1::SparseArray, a2::SparseArray)
  return SparseArraysBase.sparse_dot(a1, a2)
end

# AbstractArray interface
Base.size(a::SparseArray) = a.dims
function Base.similar(a::SparseArray, elt::Type, dims::Tuple{Vararg{Int}})
  return SparseArray{elt}(dims)
end

function Base.getindex(a::SparseArray, I...)
  return SparseArraysBase.sparse_getindex(a, I...)
end
function Base.setindex!(a::SparseArray, value, I...)
  return SparseArraysBase.sparse_setindex!(a, value, I...)
end
function Base.fill!(a::SparseArray, value)
  return SparseArraysBase.sparse_fill!(a, value)
end

# Minimal interface
SparseArraysBase.getindex_zero_function(a::SparseArray) = a.zero
SparseArraysBase.sparse_storage(a::SparseArray) = a.data
function SparseArraysBase.index_to_storage_index(
  a::SparseArray{<:Any,N}, I::CartesianIndex{N}
) where {N}
  return get(a.index_to_dataindex, I, nothing)
end
SparseArraysBase.storage_index_to_index(a::SparseArray, I) = a.dataindex_to_index[I]
function SparseArraysBase.setindex_notstored!(
  a::SparseArray{<:Any,N}, value, I::CartesianIndex{N}
) where {N}
  push!(a.data, value)
  push!(a.dataindex_to_index, I)
  a.index_to_dataindex[I] = length(a.data)
  return a
end

# TODO: Make this into a generic definition of all `AbstractArray`?
using NDTensors.SparseArraysBase: perm, stored_indices
function SparseArraysBase.stored_indices(
  a::PermutedDimsArray{<:Any,<:Any,<:Any,<:Any,<:SparseArray}
)
  return Iterators.map(
    I -> CartesianIndex(map(i -> I[i], perm(a))), stored_indices(parent(a))
  )
end

# TODO: Make this into a generic definition of all `AbstractArray`?
using NDTensors.SparseArraysBase: sparse_storage
function SparseArraysBase.sparse_storage(
  a::PermutedDimsArray{<:Any,<:Any,<:Any,<:Any,<:SparseArray}
)
  return sparse_storage(parent(a))
end

# TODO: Make this into a generic definition of all `AbstractArray`?
using NDTensors.NestedPermutedDimsArrays: NestedPermutedDimsArray
function SparseArraysBase.stored_indices(
  a::NestedPermutedDimsArray{<:Any,<:Any,<:Any,<:Any,<:SparseArray}
)
  return Iterators.map(
    I -> CartesianIndex(map(i -> I[i], perm(a))), stored_indices(parent(a))
  )
end

# TODO: Make this into a generic definition of all `AbstractArray`?
using NDTensors.NestedPermutedDimsArrays: NestedPermutedDimsArray
using NDTensors.SparseArraysBase: sparse_storage
function SparseArraysBase.sparse_storage(
  a::NestedPermutedDimsArray{<:Any,<:Any,<:Any,<:Any,<:SparseArray}
)
  return sparse_storage(parent(a))
end

# Empty the storage, helps with efficiency in `map!` to drop
# zeros.
function SparseArraysBase.dropall!(a::SparseArray)
  empty!(a.data)
  empty!(a.index_to_dataindex)
  empty!(a.dataindex_to_index)
  return a
end

# Broadcasting
function Broadcast.BroadcastStyle(arraytype::Type{<:SparseArray})
  return SparseArraysBase.SparseArrayStyle{ndims(arraytype)}()
end

# Map
function Base.map!(f, dest::AbstractArray, src::SparseArray)
  SparseArraysBase.sparse_map!(f, dest, src)
  return dest
end
function Base.copy!(dest::AbstractArray, src::SparseArray)
  SparseArraysBase.sparse_copy!(dest, src)
  return dest
end
function Base.copyto!(dest::AbstractArray, src::SparseArray)
  SparseArraysBase.sparse_copyto!(dest, src)
  return dest
end
function Base.permutedims!(dest::AbstractArray, src::SparseArray, perm)
  SparseArraysBase.sparse_permutedims!(dest, src, perm)
  return dest
end

# Base
function Base.:(==)(a1::SparseArray, a2::SparseArray)
  return SparseArraysBase.sparse_isequal(a1, a2)
end
function Base.reshape(a::SparseArray, dims::Tuple{Vararg{Int}})
  return SparseArraysBase.sparse_reshape(a, dims)
end
function Base.iszero(a::SparseArray)
  return SparseArraysBase.sparse_iszero(a)
end
function Base.isreal(a::SparseArray)
  return SparseArraysBase.sparse_isreal(a)
end
function Base.zero(a::SparseArray)
  return SparseArraysBase.sparse_zero(a)
end
function Base.one(a::SparseArray)
  return SparseArraysBase.sparse_one(a)
end
end
