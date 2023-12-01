using Dictionaries: set!
using ..SparseArrayInterface: SparseArrayInterface

SparseArrayInterface.sparse_storage(::AbstractSparseArray) = error("Not implemented")

# TODO: Move to `SparseArrayInterface`?
getindex_zero_function(::AbstractSparseArray) = error("Not implemented")

function SparseArrayInterface.index_to_storage_index(
  a::AbstractSparseArray{<:Any,N}, I::CartesianIndex{N}
) where {N}
  !isassigned(SparseArrayInterface.sparse_storage(a), I) && return nothing
  return I
end

function SparseArrayInterface.getindex_notstored(
  a::AbstractSparseArray{<:Any,N}, I::CartesianIndex{N}
) where {N}
  return getindex_zero_function(a)(a, I)
end

## # TODO: Generalize with `is_wrapped_array` trait.
## function SparseArrayInterface.getindex_notstored(
##   a::PermutedDimsArray{<:Any,N,<:AbstractSparseArray}, I::CartesianIndex{N}
## ) where {N}
##   # TODO: Need to permute `I`?
##   return getindex_zero_function(a)(a, I)
## end

function SparseArrayInterface.setindex_notstored!(
  a::AbstractSparseArray{<:Any,N}, value, I::CartesianIndex{N}
) where {N}
  set!(SparseArrayInterface.sparse_storage(a), I, value)
  return a
end
