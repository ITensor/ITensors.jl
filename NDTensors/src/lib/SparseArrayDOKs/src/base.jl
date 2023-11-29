using ..SparseArrayInterface: SparseArrayInterface

# Base
function Base.:(==)(a1::SparseArrayDOK, a2::SparseArrayDOK)
  return SparseArrayInterface.sparse_isequal(a1, a2)
end

function Base.reshape(a::SparseArrayDOK, dims::Tuple{Vararg{Int}})
  return SparseArrayInterface.sparse_reshape(a, dims)
end

function Base.zero(a::SparseArrayDOK)
  return SparseArrayInterface.sparse_zero(a)
end

function Base.one(a::SparseArrayDOK)
  return SparseArrayInterface.sparse_one(a)
end
