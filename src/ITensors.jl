
module ITensors

using Random,
      Permutations,
      Printf,
      LinearAlgebra

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

include("tagset.jl")
export addtags
include("index.jl")
export adjoint,
       dim,
       prime,
       addtags,
       settags,
       replacetags,
       removetags,
       id,
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
       swaptags
include("storage/tensorstorage.jl")
include("storage/dense.jl")
include("storage/contract.jl")
include("itensor.jl")
export svd,
       qr,
       polar,
       norm,
       commonIndex,
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
       randomMPS


end # module
