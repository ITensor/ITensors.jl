using ..SparseArraysBase: SparseArraysBase, StorageIndex, StorageIndices

SparseArraysBase.StorageIndex(i::DiagIndex) = StorageIndex(index(i))

function Base.getindex(a::AbstractDiagonalArray, i::DiagIndex)
  return a[StorageIndex(i)]
end

function Base.setindex!(a::AbstractDiagonalArray, value, i::DiagIndex)
  a[StorageIndex(i)] = value
  return a
end

SparseArraysBase.StorageIndices(i::DiagIndices) = StorageIndices(indices(i))

function Base.getindex(a::AbstractDiagonalArray, i::DiagIndices)
  return a[StorageIndices(i)]
end

function Base.setindex!(a::AbstractDiagonalArray, value, i::DiagIndices)
  a[StorageIndices(i)] = value
  return a
end
