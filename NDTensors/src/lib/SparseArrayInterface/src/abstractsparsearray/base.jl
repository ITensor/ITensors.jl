using ..SparseArrayInterface: SparseArrayInterface

# Base
function Base.:(==)(a1::SparseArrayLike, a2::SparseArrayLike)
  return SparseArrayInterface.sparse_isequal(a1, a2)
end

function Base.reshape(a::SparseArrayLike, dims::Tuple{Vararg{Int}})
  return SparseArrayInterface.sparse_reshape(a, dims)
end

function Base.zero(a::SparseArrayLike)
  return SparseArrayInterface.sparse_zero(a)
end

function Base.one(a::SparseArrayLike)
  return SparseArrayInterface.sparse_one(a)
end
