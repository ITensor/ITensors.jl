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

nstored(a::AbstractArray) = length(sparse_storage(a))

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
  return eachindex(sparse_storage(a))
end

# Derived
function index_to_storage_index(a::AbstractArray{<:Any,N}, I::Vararg{Int,N}) where {N}
  return index_to_storage_index(a, CartesianIndex(I))
end

function sparse_getindex(a::AbstractArray, I::NotStoredIndex)
  return getindex_notstored(a, index(I))
end

function sparse_getindex(a::AbstractArray, I::StoredIndex)
  return sparse_getindex(a, StorageIndex(I))
end

function sparse_getindex(a::AbstractArray, I::StorageIndex)
  return sparse_storage(a)[index(I)]
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
  return sparse_getindex(a, CartesianIndices(a)[I])
end

# Handle trailing indices
function sparse_getindex(a::AbstractArray, I::CartesianIndex)
  t = Tuple(I)
  length(t) < ndims(a) && error("Not enough indices passed")
  I′ = ntuple(i -> t[i], ndims(a))
  @assert all(i -> isone(I[i]), (ndims(a) + 1):length(I))
  return _sparse_getindex(a, CartesianIndex(I′))
end

# Update a nonzero value
function sparse_setindex!(a::AbstractArray, value, I::StorageIndex)
  sparse_storage(a)[index(I)] = value
  return a
end

# Implementation of element access
function _sparse_setindex!(a::AbstractArray{<:Any,N}, value, I::CartesianIndex{N}) where {N}
  @boundscheck checkbounds(a, I)
  sparse_setindex!(a, value, MaybeStoredIndex(a, I))
  return a
end

# Ambiguity with linear indexing
function sparse_setindex!(a::AbstractArray{<:Any,1}, value, I::CartesianIndex{1})
  _sparse_setindex!(a, value, I)
  return a
end

# Handle trailing indices or linear indexing
function sparse_setindex!(a::AbstractArray, value, I::Vararg{Int})
  sparse_setindex!(a, value, CartesianIndex(I))
  return a
end

# Linear indexing
function sparse_setindex!(a::AbstractArray, value, I::CartesianIndex{1})
  sparse_setindex!(a, value, CartesianIndices(a)[I])
  return a
end

# Handle trailing indices
function sparse_setindex!(a::AbstractArray, value, I::CartesianIndex)
  t = Tuple(I)
  length(t) < ndims(a) && error("Not enough indices passed")
  I′ = ntuple(i -> t[i], ndims(a))
  @assert all(i -> isone(I[i]), (ndims(a) + 1):length(I))
  return _sparse_setindex!(a, value, CartesianIndex(I′))
end

function sparse_setindex!(a::AbstractArray, value, I::StoredIndex)
  sparse_setindex!(a, value, StorageIndex(I))
  return a
end

function sparse_setindex!(a::AbstractArray, value, I::NotStoredIndex)
  if !iszero(value)
    setindex_notstored!(a, value, index(I))
  end
  return a
end

# A set of indices into the storage of the sparse array.
struct StorageIndices{I}
  i::I
end
indices(i::StorageIndices) = i.i

function sparse_getindex(a::AbstractArray, I::StorageIndices{Colon})
  return sparse_storage(a)
end

function sparse_getindex(a::AbstractArray, I::StorageIndices)
  return error("Not implemented")
end

function sparse_setindex!(a::AbstractArray, value, I::StorageIndices{Colon})
  sparse_storage(a) .= value
  return a
end

function sparse_setindex!(a::AbstractArray, value, I::StorageIndices)
  return error("Not implemented")
end
