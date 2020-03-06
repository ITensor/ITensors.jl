@reexport module Tensors

using StaticArrays,
      Random,
      LinearAlgebra,
      HDF5

#####################################
# DenseTensor and DiagTensor
#
include("tupletools.jl")
include("dims.jl")
include("tensorstorage.jl")
include("tensor.jl")
include("contraction_logic.jl")
include("dense.jl")
include("symmetric.jl")
include("linearalgebra.jl")
include("diag.jl")
include("combiner.jl")
include("truncate.jl")
include("svd.jl")

#####################################
# BlockSparseTensor
#
include("blocksparse/blockdims.jl")
include("blocksparse/blockoffsets.jl")
include("blocksparse/blocksparse.jl")
include("blocksparse/blocksparsetensor.jl")
include("blocksparse/diagblocksparse.jl")
include("blocksparse/combiner.jl")
include("blocksparse/linearalgebra.jl")

end # module Tensors
