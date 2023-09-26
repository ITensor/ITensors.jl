module BlockSparseArrays
using BlockArrays
using Dictionaries

export BlockSparseArray, SparseArray

include("sparsearray.jl")
include("blocksparsearray.jl")

end
