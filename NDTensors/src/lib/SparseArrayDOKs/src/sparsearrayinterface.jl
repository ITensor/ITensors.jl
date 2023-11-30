using Dictionaries: set!
using ..SparseArrayInterface: SparseArrayInterface

SparseArrayInterface.storage(a::SparseArrayDOK) = a.data

function SparseArrayInterface.index_to_storage_index(
  a::SparseArrayDOK{<:Any,N}, I::CartesianIndex{N}
) where {N}
  !isassigned(SparseArrayInterface.storage(a), I) && return nothing
  return I
end

function SparseArrayInterface.getindex_notstored(
  a::SparseArrayDOK{<:Any,N}, I::CartesianIndex{N}
) where {N}
  # TODO: Write an accessor function instead of accessing
  # with `a.zero`. Maybe `getindex_zero(a)(a, I)`
  return a.zero(a, I)
end

## # TODO: Generalize with `is_wrapped_array` trait.
## function SparseArrayInterface.getindex_notstored(
##   a::PermutedDimsArray{<:Any,N,<:SparseArrayDOK}, I::CartesianIndex{N}
## ) where {N}
##   # TODO: Write an accessor function instead of accessing
##   # with `a.zero`. Maybe `getindex_zero(a)(a, I)`
##   # TODO: Need to permute `I`?
##   return a.zero(a, I)
## end

function SparseArrayInterface.setindex_notstored!(
  a::SparseArrayDOK{<:Any,N}, value, I::CartesianIndex{N}
) where {N}
  set!(SparseArrayInterface.storage(a), I, value)
  return a
end

function SparseArrayInterface.empty_storage!(a::SparseArrayDOK)
  return empty!(SparseArrayInterface.storage(a))
end
