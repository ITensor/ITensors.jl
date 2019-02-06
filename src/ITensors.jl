
module ITensors

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
       Base.issubset,
       Base.in,
       Base.intersect,
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
       mapprime,
       swapprime,
       plev,
       tags,
       rank,
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
       difference,
       primeexcept,
       swaptags
include("storage/tensorstorage.jl")
include("storage/dense.jl")
include("storage/contract.jl")
include("itensor.jl")
export svd,
       qr,
       polar,
       norm,
       commonindex,
       commoninds,
       delta,
       δ

include("decomp.jl")
#export truncate!

include("mps/siteset.jl")
export SiteSet,
       Sites

include("mps/initstate.jl")
export InitState

include("mps/mps.jl")
export MPS,
       position!,
       overlap,
       randomMPS


end # module
