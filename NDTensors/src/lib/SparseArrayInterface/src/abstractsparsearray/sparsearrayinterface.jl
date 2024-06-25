using Dictionaries: set!
using ..SparseArrayInterface: SparseArrayInterface

SparseArrayInterface.sparse_storage(::AbstractSparseArray) = error("Not implemented")

function SparseArrayInterface.index_to_storage_index(
  a::AbstractSparseArray{<:Any,N}, I::CartesianIndex{N}
) where {N}
  !isassigned(SparseArrayInterface.sparse_storage(a), I) && return nothing
  return I
end

function SparseArrayInterface.setindex_notstored!(
  a::AbstractSparseArray{<:Any,N}, value, I::CartesianIndex{N}
) where {N}
  iszero(value) && return a
  return error("Setting the specified unstored index is not supported.")
end

# TODO: Check if this is efficient, or determine if this mapping should
# be performed in `storage_index_to_index` and/or `index_to_storage_index`.
function SparseArrayInterface.sparse_storage(a::SubArray{<:Any,<:Any,<:AbstractSparseArray})
  parent_storage = sparse_storage(parent(a))
  all_sliced_storage_indices = map(keys(parent_storage)) do I
    return map_index(a.indices, I)
  end
  sliced_storage_indices = filter(!isnothing, all_sliced_storage_indices)
  sliced_parent_storage = map(I -> parent_storage[I], keys(sliced_storage_indices))
  return typeof(parent_storage)(sliced_storage_indices, sliced_parent_storage)
end

function SparseArrayInterface.stored_indices(
  a::PermutedDimsArray{<:Any,<:Any,<:Any,<:Any,<:AbstractSparseArray}
)
  return Iterators.map(
    I -> CartesianIndex(map(i -> I[i], perm(a))), stored_indices(parent(a))
  )
end

function SparseArrayInterface.sparse_storage(
  a::PermutedDimsArray{<:Any,<:Any,<:Any,<:Any,<:AbstractSparseArray}
)
  return sparse_storage(parent(a))
end
