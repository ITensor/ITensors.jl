module BlockSparseArrays

# factorizations
include("factorizations/svd.jl")

# possible upstream contributions
include("BlockArraysExtensions/BlockArraysExtensions.jl")

# interface functions that don't have to specialize
include("blocksparsearrayinterface/blocksparsearrayinterface.jl")
include("blocksparsearrayinterface/linearalgebra.jl")
include("blocksparsearrayinterface/blockzero.jl")
include("blocksparsearrayinterface/broadcast.jl")
include("blocksparsearrayinterface/map.jl")
include("blocksparsearrayinterface/arraylayouts.jl")
include("blocksparsearrayinterface/views.jl")

# functions defined for any abstractblocksparsearray
include("abstractblocksparsearray/abstractblocksparsearray.jl")
include("abstractblocksparsearray/wrappedabstractblocksparsearray.jl")
include("abstractblocksparsearray/abstractblocksparsematrix.jl")
include("abstractblocksparsearray/abstractblocksparsevector.jl")
include("abstractblocksparsearray/views.jl")
include("abstractblocksparsearray/arraylayouts.jl")
include("abstractblocksparsearray/sparsearrayinterface.jl")
include("abstractblocksparsearray/broadcast.jl")
include("abstractblocksparsearray/map.jl")
include("abstractblocksparsearray/linearalgebra.jl")

# functions specifically for BlockSparseArray
include("blocksparsearray/defaults.jl")
include("blocksparsearray/blocksparsearray.jl")
include("blocksparsearray/blockdiagonalarray.jl")

include("BlockArraysSparseArrayInterfaceExt/BlockArraysSparseArrayInterfaceExt.jl")
include("../ext/BlockSparseArraysTensorAlgebraExt/src/BlockSparseArraysTensorAlgebraExt.jl")
include("../ext/BlockSparseArraysGradedAxesExt/src/BlockSparseArraysGradedAxesExt.jl")
include("../ext/BlockSparseArraysAdaptExt/src/BlockSparseArraysAdaptExt.jl")
end
