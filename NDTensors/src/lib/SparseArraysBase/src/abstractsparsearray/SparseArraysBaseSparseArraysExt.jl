using Base: Forward
using SparseArrays: SparseArrays, SparseMatrixCSC, findnz, getcolptr, nonzeros, rowvals
using ..SparseArraysBase: stored_length

# Julia Base `AbstractSparseArray` interface
SparseArrays.nnz(a::AbstractSparseArray) = stored_length(a)

sparse_storage(a::SparseMatrixCSC) = nonzeros(a)
function storage_index_to_index(a::SparseMatrixCSC, I)
  I1s, I2s = findnz(a)
  return CartesianIndex(I1s[I], I2s[I])
end
function index_to_storage_index(a::SparseMatrixCSC, I::CartesianIndex{2})
  i0, i1 = Tuple(I)
  r1 = getcolptr(a)[i1]
  r2 = getcolptr(a)[i1 + 1] - 1
  (r1 > r2) && return nothing
  r1 = searchsortedfirst(rowvals(a), i0, r1, r2, Forward)
  return ((r1 > r2) || (rowvals(a)[r1] != i0)) ? nothing : r1
end
