module BlockSparseArrays
using BlockArrays
using Compat
using Dictionaries
using SplitApplyCombine
using LinearAlgebra: Hermitian, Transpose

using BlockArrays: block

export BlockSparseArray, SparseArray

include("tensor_product.jl")
include("base.jl")
include("axes.jl")
include("abstractarray.jl")
include("permuteddimsarray.jl")
include("hermitian.jl")
include("transpose.jl")
include("blockarrays.jl")
# TODO: Split off into `NDSparseArrays` module.
include("sparsearray.jl")
include("blocksparsearray.jl")
include("allocate_output.jl")
include("subarray.jl")
include("broadcast.jl")
include("fusedims.jl")
include("gradedrange.jl")

end
