module ITensors

using Random,
      Printf,
      LinearAlgebra,
      StaticArrays,
      TimerOutputs,
      Reexport,
      HDF5

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
       Base.lastindex,
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
#
const GLOBAL_PARAMS = Dict("WarnTensorOrder" => 14)
const GLOBAL_TIMER = TimerOutput()

#####################################
# Tensor
#
include("Tensors/Tensors.jl")

#####################################
# Index and IndexSet
#
include("smallstring.jl")
include("readwrite.jl")
include("tagset.jl")
include("index.jl")
include("indexset.jl")

#####################################
# ITensor
#
include("itensor.jl")
include("decomp.jl")
include("iterativesolvers.jl")

###########################################################
# MPS/MPO
#
include("mps/mps.jl")
include("mps/mpo.jl")
include("mps/sweeps.jl")
include("mps/projmpo.jl")
include("mps/observer.jl")
include("mps/dmrg.jl")

###########################################################
# Physics
#
include("physics/tag_types.jl")
include("physics/lattices.jl")
include("physics/qn.jl")
include("physics/site_types/spinhalf.jl")
include("physics/site_types/spinone.jl")
include("physics/site_types/electron.jl")
include("physics/site_types/tj.jl")
include("physics/autompo.jl")

end # module ITensors
