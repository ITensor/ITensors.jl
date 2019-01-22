
module ITensors

using Random,
      Permutations,
      LinearAlgebra

import Base.length,
       Base.getindex,
       Base.setindex!,
       Base.convert,
       Base.==,
       Base.!=,
       Base.isapprox,
       Base.+,
       Base.-,
       Base.*,
       Base.copy,
       Base.push!,
       Base.iterate,
       Base.size,
       Base.conj,
       Random.randn!,
       Base.eltype,
       Base.issubset,
       LinearAlgebra.norm,
       LinearAlgebra.svd

## Types
export Dense,
       TagSet,
       Index,
       IndexVal,
       IndexSet,
       ITensor,
       In,
       Out

## Functions
export prime!,
       rank,
       order,
       dims,
       randomITensor,
       inds,
       scalar,
       permute,
       store,
       data,
       dag,
       dir,
       val

include("tagset.jl")
include("index.jl")
export dim,
       prime,
       settags,
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
       difference
include("storage/tensorstorage.jl")
include("storage/dense.jl")
include("storage/contract.jl")
include("itensor.jl")
export svd

end # module
