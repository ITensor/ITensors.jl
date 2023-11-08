module BlockSparseArrays
using BlockArrays
using Compat
using Dictionaries

using BlockArrays: block

export BlockSparseArray, SparseArray

include("abstractarray.jl")
include("permuteddimsarray.jl")
include("sparsearray.jl")
include("blocksparsearray.jl")
include("broadcast.jl")

end
