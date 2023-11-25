module LinearAlgebraExt
using ...AlgorithmSelection: Algorithm, @Algorithm_str
using BlockArrays: BlockArrays, blockedrange, blocks
using ..BlockSparseArrays: SparseArray, nonzero_keys # TODO: Move to `SparseArraysExtensions` module, rename `SparseArrayDOK`.
using ..BlockSparseArrays: BlockSparseArrays, BlockSparseArray, nonzero_blockkeys
using LinearAlgebra: LinearAlgebra, Hermitian, Transpose, I, eigen, qr
using SparseArrays: SparseArrays, SparseMatrixCSC, spzeros, sparse

# TODO: Move to `SparseArraysExtensions`.
include("hermitian.jl")
# TODO: Move to `SparseArraysExtensions`.
include("transpose.jl")
include("qr.jl")
include("eigen.jl")
include("svd.jl")
end
