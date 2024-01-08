using SparseArrays: SparseArrays, SparseMatrixCSC
using ..SparseArrayInterface: nstored

# Julia Base `AbstractSparseArray` interface
SparseArrays.nnz(a::AbstractSparseArray) = nstored(a)

sparse_storage(a::SparseMatrixCSC) = error("Not implemented")
storage_index_to_index(a::SparseMatrixCSC, I) = error("Not implemented")
index_to_storage_index(a::SparseMatrixCSC{<:Any,N}, I::CartesianIndex{N}) where {N} = error("Not implemented")
