
module ITensors

using Random,
      Permutations,
      LinearAlgebra

import Base.length,
       Base.getindex,
       Base.setindex!,
       Base.convert,
       Base.==,
       Base.+,
       Base.-,
       Base.*,
       Base.copy,
       Base.push!,
       Base.iterate,
       Base.size,
       Base.conj,
       Random.randn!,
       LinearAlgebra.norm

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
export prime,
       prime!,
       rank,
       order,
       dim,
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
include("indexset.jl")
include("storage/tensorstorage.jl")
include("storage/dense.jl")
include("storage/contract.jl")
include("itensor.jl")

end # module
