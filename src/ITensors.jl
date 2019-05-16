
module ITensors

const Number64 = Union{Float64,ComplexF64}

using Random,
      Permutations,
      Printf,
      LinearAlgebra,
      StaticArrays # For SmallString

import Base.adjoint,
       Base.length,
       Base.getindex,
       Base.setindex!,
       Base.convert,
       Base.==,
       Base.!=,
       Base.isapprox,
       Base.+,
       Base.-,
       Base.*,
       Base./,
       Base.isless,
       Base.copy,
       Base.push!,
       Base.iterate,
       Base.size,
       Base.show,
       Base.conj,
       Random.randn!,
       Base.eltype,
       LinearAlgebra.norm,
       LinearAlgebra.svd,
       LinearAlgebra.qr,
       LinearAlgebra.eigen

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
export Site,
       BasicSite,
       SiteSet,
       ind,
       op
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
       overlap,
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

include("mps/dmrg.jl")
export dmrg!

end # module
