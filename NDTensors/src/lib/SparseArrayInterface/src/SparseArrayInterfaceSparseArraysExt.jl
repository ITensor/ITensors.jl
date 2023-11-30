## using SparseArrays: SparseArrays, SparseMatrixCSC
## 
## # Julia Base `SparseArrays.AbstractSparseArray` interface
## # SparseMatrixCSC.nnz
## nnz(a::AbstractArray) = nonzero_length(a)
## 
## # SparseArrayInterface.SparseMatrixCSC
## function SparseMatrixCSC(a::AbstractMatrix)
##   return error("Not implemented")
## end
