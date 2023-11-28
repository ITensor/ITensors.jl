# An index into the storage of the sparse array.
struct StorageIndex{I}
  i::I
end
index(i::StorageIndex) = i.i

# Indicate if the index into the sparse array is
# stored or not.
abstract type MaybeStoredIndex{I} end

# An index into a stored value of the sparse array.
# Stores both the index into the outer array
# as well as into the underlying storage.
struct StoredIndex{Iouter,Istorage} <: MaybeStoredIndex{Iouter}
  iouter::Iouter
  istorage::StorageIndex{Istorage}
end
index(i::StoredIndex) = i.iouter
StorageIndex(i::StoredIndex) = i.istorage

nstored(a::AbstractArray) = length(storage(a))

struct NotStoredIndex{Iouter} <: MaybeStoredIndex{Iouter}
  iouter::Iouter
end
index(i::NotStoredIndex) = i.iouter

function MaybeStoredIndex(a::AbstractArray, I)
  return MaybeStoredIndex(I, index_to_storage_index(a, I))
end
MaybeStoredIndex(I, I_storage) = StoredIndex(I, StorageIndex(I_storage))
MaybeStoredIndex(I, I_storage::Nothing) = NotStoredIndex(I)

function storage_indices(a::AbstractArray)
  return eachindex(storage(a))
end

# Derived
function index_to_storage_index(a::AbstractArray{<:Any,N}, I::Vararg{Int,N}) where {N}
  return index_to_storage_index(a, CartesianIndex(I))
end

# Helper type for constructing zero values
struct GetIndexZero end
(::GetIndexZero)(a::AbstractArray, I) = zero(eltype(a))

function sparse_getindex(
  a::AbstractArray{<:Any,N}, I::NotStoredIndex{CartesianIndex{N}}
) where {N}
  return getindex_notstored(a, index(I))
end

function sparse_getindex(
  a::AbstractArray{<:Any,N}, I::StoredIndex{CartesianIndex{N}}
) where {N}
  return sparse_getindex(a, StorageIndex(I))
end

function sparse_getindex(a::AbstractArray, I::StorageIndex)
  return storage(a)[index(I)]
end

function sparse_getindex(a::AbstractArray{<:Any,N}, I::Vararg{Int,N}) where {N}
  return sparse_getindex(a, CartesianIndex(I))
end

function sparse_getindex(a::AbstractArray{<:Any,N}, I::CartesianIndex{N}) where {N}
  return _sparse_getindex(a, I)
end

# Ambiguity with linear indexing
function sparse_getindex(a::AbstractArray{<:Any,1}, I::CartesianIndex{1})
  return _sparse_getindex(a, I)
end

# Implementation of element access
function _sparse_getindex(a::AbstractArray{<:Any,N}, I::CartesianIndex{N}) where {N}
  @boundscheck checkbounds(a, I)
  return sparse_getindex(a, MaybeStoredIndex(a, I))
end

# Handle trailing indices or linear indexing
function sparse_getindex(a::AbstractArray, I::Vararg{Int})
  return sparse_getindex(a, CartesianIndex(I))
end

# Linear indexing
function sparse_getindex(a::AbstractArray, I::CartesianIndex{1})
  return error("Linear indexing not supported yet for sparse arrays")
end

# Handle trailing indices
function sparse_getindex(a::AbstractArray, I::CartesianIndex)
  t = Tuple(I)
  length(t) < ndims(a) && error("Not enough indices passed")
  I′ = ntuple(i -> t[i], ndims(a))
  @assert all(i -> isone(I[i]), (ndims(a) + 1):length(I))
  return _sparse_getindex(a, I′)
end

# Update a nonzero value
function sparse_setindex!(a::AbstractArray, value, I::StorageIndex)
  storage(a)[index(I)] = value
  return a
end

function sparse_setindex!(a::AbstractArray{<:Any,N}, value, I::Vararg{Int,N}) where {N}
  sparse_setindex!(a, value, CartesianIndex(I))
  return a
end

function sparse_setindex!(
  a::AbstractArray{<:Any,N}, value, I::StoredIndex{CartesianIndex{N}}
) where {N}
  sparse_setindex!(a, value, StorageIndex(I))
  return a
end

function sparse_setindex!(
  a::AbstractArray{<:Any,N}, value, I::NotStoredIndex{CartesianIndex{N}}
) where {N}
  if !iszero(value)
    setindex_notstored!(a, value, index(I))
  end
  return a
end

function sparse_setindex!(a::AbstractArray{<:Any,N}, value, I::CartesianIndex{N}) where {N}
  sparse_setindex!(a, value, MaybeStoredIndex(a, I))
  return a
end
