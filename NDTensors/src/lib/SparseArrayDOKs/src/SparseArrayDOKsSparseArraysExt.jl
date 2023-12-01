using SparseArrays: SparseArrays
using ..SparseArrayInterface: nstored

# Julia Base `AbstractSparseArray` interface
SparseArrays.nnz(a::AbstractSparseArray) = nstored(a)
