module LinearAlgebraExt
using BlockArrays: BlockArrays, blockedrange, blocks
using ..BlockSparseArrays: SparseArray, nonzero_keys # TODO: Move to `SparseArraysExtensions` module, rename `SparseArrayDOK`.
using ..BlockSparseArrays: BlockSparseArrays, BlockSparseArray, nonzero_blockkeys
using LinearAlgebra: LinearAlgebra, Hermitian, Transpose, I, qr
using ...NDTensors: Algorithm, @Algorithm_str # TODO: Move to `AlgorithmSelector` module.
using SparseArrays: SparseArrays, SparseMatrixCSC, spzeros, sparse

# TODO: Move to `SparseArraysExtensions`.
include("hermitian.jl")
# TODO: Move to `SparseArraysExtensions`.
include("transpose.jl")
include("qr.jl")
end
