
module ITensors

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
       swaptags,
       commoninds,
       commonindex,
       uniqueinds,
       uniqueindex
include("storage/tensorstorage.jl")
include("storage/dense.jl")
include("storage/contract.jl")
include("itensor.jl")
export svd,
       qr,
       polar,
       eigen,
       norm,
       findindex,
       commonindex,
       findtags,
       commoninds,
       delta,
       δ,
       isNull

include("decomp.jl")
#export truncate!

include("mps/siteset.jl")
export BasicSite,
       Site,
       SiteSet,
       ind,
       op,
       replaceBond!

include("mps/sitesets/spinhalf.jl")
export SpinHalfSite,
       spinhalfs
include("mps/sitesets/spinone.jl")
export SpinOneSite,
       spinones
include("mps/sitesets/electron.jl")
export ElectronSite,
       electrons
include("mps/sitesets/tj.jl")
export tJSite,
       tjs

include("mps/initstate.jl")
export InitState

include("mps/mps.jl")
export MPS,
       position!,
       inner,
       randomMPS

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
       LProj,RProj
include("mps/dmrg.jl")
export dmrg

# Development folder includes helper
# codes not intended for final release,
# just to ease development temporarily
include("development/heisenberg.jl")
export setElt,
       Heisenberg

end # module
