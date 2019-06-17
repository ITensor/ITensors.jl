
module ITensors

function pause() 
  println("(paused)")
  readline(stdin)
end

using Random,
      Permutations,
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
       Base.isapprox,
       Base.isless,
       Base.iterate,
       Base.length,
       Base.push!,
       Base.setindex!,
       Base.show,
       Base.similar,
       Base.size,
       Base.==,
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
       LinearAlgebra.eigen,
       LinearAlgebra.norm,
       LinearAlgebra.mul!,
       LinearAlgebra.svd,
       LinearAlgebra.rmul!,
       LinearAlgebra.qr,
       Random.randn!

## Types
export Dense,
       TagSet,
       Index,
       IndexVal,
       IndexSet,
       ITensor,
       In,
       Out,
       Neither

## Functions
export prime,
       setprime,
       noprime,
       plev,
       tags,
       order,
       dims,
       randomITensor,
       id,
       inds,
       scalar,
       permute,
       store,
       data,
       dag,
       dir,
       sim,
       val

#TODO: continue work on SmallString, use as Tags
include("smallstring.jl")
include("tagset.jl")
export addtags,
       hastags,
       Tag
include("index.jl")
export adjoint,
       dim,
       prime,
       addtags,
       settags,
       replacetags,
       removetags,
       hastags,
       id,
       isdefault,
       dir,
       plev,
       tags,
       ind,
       Neither,
       In,
       Out
include("indexset.jl")
export hasindex,
       hasinds,
       hassameinds,
       findindex,
       findinds,
       swaptags,
       commoninds,
       commonindex,
       uniqueinds,
       uniqueindex
include("storage/tensorstorage.jl")
include("storage/dense.jl")
include("storage/contract.jl")
export CProps, contract!, compute!, compute_contraction_labels, contract_inds, contract
include("itensor.jl")
export svd,
       qr,
       polar,
       eigen,
       norm,
       delta,
       δ,
       isNull

include("decomp.jl")
#export truncate!

include("iterativesolvers.jl")
export davidson

include("mps/siteset.jl")
export BasicSite,
       Site,
       SiteSet,
       ind,
       op,
       replaceBond!

include("mps/sitesets/spinhalf.jl")
export SpinHalfSite,
       spinHalfSites
include("mps/sitesets/spinone.jl")
export SpinOneSite,
       spinOneSites
include("mps/sitesets/electron.jl")
export ElectronSite,
       electronSites
include("mps/sitesets/tj.jl")
export tJSite,
       tjSites

include("mps/initstate.jl")
export InitState

include("mps/mps.jl")
export MPS,
       position!,
       inner,
       randomMPS,
       maxDim,
       linkindex,
       siteindex

include("mps/mpo.jl")
export MPO

include("mps/sweeps.jl")
export Sweeps,
       nsweep,
       maxdim,
       mindim,
       cutoff,
       maxdim!,
       mindim!,
       cutoff!,
       sweepnext


include("mps/projmpo.jl")
export ProjMPO,
       LProj,
       RProj,
       product

include("mps/dmrg.jl")
export dmrg

include("mps/autompo.jl")
export AutoMPO,
       add!

# Development folder includes helper
# codes not intended for final release,
# just to ease development temporarily
include("development/heisenberg.jl")
export setElt,
       Heisenberg

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
