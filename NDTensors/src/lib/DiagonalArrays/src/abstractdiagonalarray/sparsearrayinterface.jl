using Compat: Returns, allequal
using ..SparseArrayInterface: SparseArrayInterface

# `SparseArrayInterface` interface
function SparseArrayInterface.index_to_storage_index(
  a::AbstractDiagonalArray{<:Any,N}, I::CartesianIndex{N}
) where {N}
  !allequal(Tuple(I)) && return nothing
  return first(Tuple(I))
end

function SparseArrayInterface.storage_index_to_index(a::AbstractDiagonalArray, I)
  return CartesianIndex(ntuple(Returns(I), ndims(a)))
end

## # 1-dimensional case can be `AbstractDiagonalArray`.
## function SparseArrayInterface.sparse_similar(
##   a::AbstractDiagonalArray, elt::Type, dims::Tuple{Int}
## )
##   # TODO: Handle preserving zero element function.
##   return similar(a, elt, dims)
## end
