using ..SparseArrayInterface: SparseArrayInterface

# Base
function Base.:(==)(a1::AbstractSparseArray, a2::AbstractSparseArray)
  return SparseArrayInterface.sparse_isequal(a1, a2)
end

function Base.reshape(a::AbstractSparseArray, dims::Tuple{Vararg{Int}})
  return SparseArrayInterface.sparse_reshape(a, dims)
end

function Base.zero(a::AbstractSparseArray)
  return SparseArrayInterface.sparse_zero(a)
end

function Base.one(a::AbstractSparseArray)
  return SparseArrayInterface.sparse_one(a)
end
