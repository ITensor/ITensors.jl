
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
include("mps/sitesets/spinhalf.jl")
include("mps/sitesets/spinone.jl")
include("mps/sitesets/electron.jl")
include("mps/sitesets/tj.jl")
include("mps/initstate.jl")
include("mps/mps.jl")
include("mps/mpo.jl")
include("mps/sweeps.jl")
include("mps/projmpo.jl")
include("mps/dmrg.jl")
include("mps/autompo.jl")

mutable struct Timers
  contract_t::Float64
  contract_c::Int
  gemm_t::Float64
  gemm_c::Int
  permute_t::Float64
  permute_c::Int
  svd_t::Float64
  svd_c::Int
  svd_store_t::Float64
  svd_store_c::Int
  svd_store_svd_t::Float64
  svd_store_svd_c::Int
end

timer = Timers(0.0,0,0,0.0,0.0,0,0.0,0,0.0,0,0.0,0)
export timer

function reset!(t::Timers)
  t.contract_t = 0.0; t.contract_c = 0;
  t.gemm_t = 0.0; t.gemm_c = 0;
  t.permute_t = 0.0; t.permute_c = 0;
  t.svd_t = 0.0; t.svd_c = 0;
  t.svd_store_t = 0.0; t.svd_store_c = 0;
  t.svd_store_svd_t = 0.0; t.svd_store_svd_c = 0;
end
export reset!

function printTimes(t::Timers)
  @printf "contract_t = %.6f (%d), Total = %.4f \n" (t.contract_t/t.contract_c) t.contract_c t.contract_t
  @printf "  gemm_t = %.6f (%d), Total = %.4f \n" (t.gemm_t/t.gemm_c) t.gemm_c t.gemm_t
  @printf "  permute_t = %.6f (%d), Total = %.4f \n" (t.permute_t/t.permute_c) t.permute_c t.permute_t
  @printf "svd_t = %.6f (%d), Total = %.4f \n" (t.svd_t/t.svd_c) t.svd_c t.svd_t
  @printf "  svd_store_t = %.6f (%d), Total = %.4f \n" (t.svd_store_t/t.svd_store_c) t.svd_store_c t.svd_store_t
  @printf "  svd_store_svd_t = %.6f (%d), Total = %.4f \n" (t.svd_store_svd_t/t.svd_store_svd_c) t.svd_store_svd_c t.svd_store_svd_t
end
export printTimes


end # module
