module ITensors

using Random,
      Printf,
      LinearAlgebra,
      StaticArrays # For SmallString

# TODO: move imports to individual files
import Base.adjoint,
       Base.conj,
       Base.convert,
       Base.copy,
       Base.deepcopy,
       Base.copyto!,
       Base.eltype,
       Base.fill!,
       Base.getindex,
       Base.in,
       Base.isapprox,
       Base.isless,
       Base.iterate,
       Base.length,
       Base.push!,
       Base.setindex!,
       Base.eachindex,
       Base.show,
       Base.sum,
       Base.summary,
       Base.similar,
       Base.size,
       Base.ndims,
       Base.!=,
       Base.+,
       Base.-,
       Base.*,
       Base./,
       Base.^,
       Base.setdiff,  # Since setdiff doesn't 
                      # work with IndexSet, overload it
       LinearAlgebra.axpby!,
       LinearAlgebra.axpy!,
       LinearAlgebra.dot,
       LinearAlgebra.norm,
       LinearAlgebra.mul!,
       LinearAlgebra.rmul!,
       LinearAlgebra.normalize!,
       Random.randn!


#####################################
# Global Variables

const warnTensorOrder = 10

#####################################
# Helper functionality
#

include("sets.jl")

#####################################

include("smallstring.jl")
include("tagset.jl")
include("index.jl")
include("indexset.jl")
include("tensor/tensor.jl")
include("tensor/tensorstorage.jl")
include("tensor/dense.jl")
include("tensor/diag.jl")
include("tensor/combiner.jl")
include("tensor/contract.jl")
include("tensor/svd.jl")
include("itensor.jl")
include("decomp.jl")
include("iterativesolvers.jl")

###########################################################
# MPS/MPO
#
include("mps/siteset.jl")
include("mps/mps.jl")
include("mps/mpo.jl")
include("mps/sweeps.jl")
include("mps/projmpo.jl")
include("mps/dmrg.jl")
include("mps/autompo.jl")

###########################################################
# Physics
#
include("physics/lattices.jl")
include("physics/sitesets/spinhalf.jl")
include("physics/sitesets/spinone.jl")
include("physics/sitesets/electron.jl")
include("physics/sitesets/tj.jl")

end # module
