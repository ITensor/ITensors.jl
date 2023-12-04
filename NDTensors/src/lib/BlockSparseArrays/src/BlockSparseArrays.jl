module BlockSparseArrays
include("blocksparsearrayinterface/blocksparsearrayinterface.jl")
include("blocksparsearrayinterface/linearalgebra.jl")
include("blocksparsearrayinterface/blockzero.jl")
include("abstractblocksparsearray/abstractblocksparsearray.jl")
include("abstractblocksparsearray/abstractblocksparsematrix.jl")
include("abstractblocksparsearray/abstractblocksparsevector.jl")
include("abstractblocksparsearray/arraylayouts.jl")
include("abstractblocksparsearray/linearalgebra.jl")
include("blocksparsearray/defaults.jl")
include("blocksparsearray/blocksparsearray.jl")
include("BlockArraysExtensions/BlockArraysExtensions.jl")
end
