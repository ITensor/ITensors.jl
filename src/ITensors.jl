module ITensors

#####################################
# Tensors
#
include("Tensors/Tensors.jl")

using Random
using Printf
using LinearAlgebra
using StaticArrays
using TimerOutputs
using HDF5
using KrylovKit
using .Tensors

#####################################
# Exports from Tensors module
#
export truncerror,
       Spectrum,
       eigs,
       entropy

#####################################
# Global Variables
#
const GLOBAL_PARAMS = Dict("WarnTensorOrder" => 14)
const GLOBAL_TIMER = TimerOutput()

#####################################
# Index and IndexSet
#
include("smallstring.jl")
include("readwrite.jl")
include("not.jl")
include("tagset.jl")
include("arrow.jl")
include("index.jl")
include("indexset.jl")

#####################################
# ITensor
#
include("itensor.jl")
include("broadcast.jl")
include("decomp.jl")
include("iterativesolvers.jl")

#####################################
# QNs
#
include("qn.jl")
include("qnindex.jl")
include("qnitensor.jl")

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
include("physics/site_types/spinhalf.jl")
include("physics/site_types/spinone.jl")
include("physics/site_types/fermion.jl")
include("physics/site_types/electron.jl")
include("physics/site_types/tj.jl")
include("physics/autompo.jl")

include("developer_tools.jl")

end # module ITensors
