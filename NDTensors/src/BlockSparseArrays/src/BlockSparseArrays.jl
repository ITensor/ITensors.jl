module BlockSparseArrays
using BlockArrays
using Compat
using Dictionaries
using SplitApplyCombine

using BlockArrays: block

export BlockSparseArray, SparseArray

include("tensor_product.jl")
include("base.jl")
include("axes.jl")
include("abstractarray.jl")
include("permuteddimsarray.jl")
include("blockarrays.jl")
include("sparsearray.jl")
include("blocksparsearray.jl")
include("allocate_output.jl")
include("subarray.jl")
include("broadcast.jl")
include("fusedims.jl")
include("gradedrange.jl")

end
