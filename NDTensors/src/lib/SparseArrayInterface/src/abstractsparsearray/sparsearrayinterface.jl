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
  set!(SparseArrayInterface.sparse_storage(a), I, value)
  return a
end
