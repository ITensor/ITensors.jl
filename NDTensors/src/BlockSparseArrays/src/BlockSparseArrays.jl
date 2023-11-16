module BlockSparseArrays
## using Compat
## using SplitApplyCombine
using BlockArrays:
  BlockArrays,
  AbstractBlockArray,
  Block,
  BlockIndex,
  BlockRange,
  BlockedUnitRange,
  findblockindex,
  block,
  blockcheckbounds,
  blocklength,
  blockedrange,
  blocks
using Dictionaries: Dictionary, set! # TODO: Move to `SparseArraysExtensions`.
using LinearAlgebra: Hermitian

export BlockSparseArray, SparseArray

include("tensor_product.jl")
include("base.jl")
include("axes.jl")
include("abstractarray.jl")
include("permuteddimsarray.jl")
include("blockarrays.jl")
# TODO: Split off into `SparseArraysExtensions` module, rename to `SparseArrayDOK`.
include("sparsearray.jl")
include("blocksparsearray.jl")
include("allocate_output.jl")
include("subarray.jl")
include("broadcast.jl")
include("fusedims.jl")
include("gradedrange.jl")
include("LinearAlgebraExt/LinearAlgebraExt.jl")

end
