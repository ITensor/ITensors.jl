
module ITensors

function pause() 
  println("(paused)")
  readline(stdin)
end

using Random,
      Printf,
      LinearAlgebra,
      StaticArrays # For SmallString

import Base.adjoint,
       Base.conj,
       Base.convert,
       Base.copy,
       Base.copyto!,
       Base.eltype,
       Base.fill!,
       Base.getindex,
       Base.in,
       Base.isapprox,
       Base.isless,
       Base.iterate,
       Base.length,
       Base.ndims,
       Base.push!,
       Base.setindex!,
       Base.show,
       Base.similar,
       Base.size,
       Base.!=,
       Base.+,
       Base.-,
       Base.*,
       Base./,
       Base.^,
       Base.complex,
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


#TODO: continue work on SmallString, use as Tags
include("smallstring.jl")
include("tagset.jl")
include("index.jl")
include("indexset.jl")
include("storage/tensorstorage.jl")
include("storage/dense.jl")
include("storage/contract.jl")
include("storage/svd.jl")
#export CProps, contract!, compute!, compute_contraction_labels, contract_inds, contract
include("itensor.jl")
include("decomp.jl")
include("iterativesolvers.jl")

###########################################################
# MPS/MPO
#
include("mps/siteset.jl")
include("mps/initstate.jl")
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

using Pkg
if "CuArrays" ∈ keys(Pkg.installed())
    include("CuITensors.jl")
    export cuITensor,
           randomCuITensor,
           cuMPS,
           randomCuMPS
end

end # module
