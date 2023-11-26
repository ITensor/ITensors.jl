using SparseArrays: SparseArrays

# Julia Base `AbstractSparseArray` interface
SparseArrays.nnz(a::SparseArrayDOK) = nonzero_length(a)

