module BlockSparseArrays
using BlockArrays
using Compat
using Dictionaries

using BlockArrays: block

export BlockSparseArray, SparseArray

include("base.jl")
include("axes.jl")
include("abstractarray.jl")
include("permuteddimsarray.jl")
include("blockarrays.jl")
include("sparsearray.jl")
include("blocksparsearray.jl")
include("subarray.jl")
include("broadcast.jl")
include("gradedblockedunitrange.jl")
include("fusedims.jl")

end
