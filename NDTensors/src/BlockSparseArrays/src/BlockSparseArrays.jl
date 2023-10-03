module BlockSparseArrays
using BlockArrays
using Dictionaries

using BlockArrays: block

export BlockSparseArray, SparseArray

include("sparsearray.jl")
include("blocksparsearray.jl")

end
