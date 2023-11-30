using Dictionaries: set!
using ..SparseArrayInterface: SparseArrayInterface

SparseArrayInterface.storage(a::SparseArrayDOK) = a.data

function SparseArrayInterface.index_to_storage_index(
  a::SparseArrayDOK{<:Any,N}, I::CartesianIndex{N}
) where {N}
  !isassigned(SparseArrayInterface.storage(a), I) && return nothing
  return I
end

function SparseArrayInterface.setindex_notstored!(
  a::SparseArrayDOK{<:Any,N}, value, I::CartesianIndex{N}
) where {N}
  set!(SparseArrayInterface.storage(a), I, value)
  return a
end

function SparseArrayInterface.empty_storage!(a::SparseArrayDOK)
  return empty!(SparseArrayInterface.storage(a))
end
