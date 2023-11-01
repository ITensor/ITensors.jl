module BlockSparseArrays
using BlockArrays
using Dictionaries

using BlockArrays: block

export BlockSparseArray, SparseArray

include("abstractarray.jl")
include("permuteddimsarray.jl")
include("sparsearray.jl")
include("blocksparsearray.jl")

end
