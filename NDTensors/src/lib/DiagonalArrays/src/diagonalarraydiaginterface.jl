using ..SparseArrayInterface: SparseArrayInterface, StorageIndex, StorageIndices

SparseArrayInterface.StorageIndex(i::DiagIndex) = StorageIndex(index(i))

function Base.getindex(a::DiagonalArray, i::DiagIndex)
  return a[StorageIndex(i)]
end

function Base.setindex!(a::DiagonalArray, value, i::DiagIndex)
  a[StorageIndex(i)] = value
  return a
end

SparseArrayInterface.StorageIndices(i::DiagIndices) = StorageIndices(indices(i))

function Base.getindex(a::DiagonalArray, i::DiagIndices)
  return a[StorageIndices(i)]
end

function Base.setindex!(a::DiagonalArray, value, i::DiagIndices)
  a[StorageIndices(i)] = value
  return a
end
