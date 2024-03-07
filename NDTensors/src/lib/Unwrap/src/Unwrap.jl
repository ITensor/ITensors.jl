module Unwrap
using SimpleTraits
using LinearAlgebra
using Base: ReshapedArray
using StridedViews
using Adapt: Adapt, adapt, adapt_structure

include("expose.jl")

include("import.jl")
## TODO Create functions which take the `Expose` type and launch functions
## using that type
## Exposed based functions
include("functions/abstractarray.jl")
include("functions/copyto.jl")
include("functions/linearalgebra.jl")
include("functions/mul.jl")
include("functions/permutedims.jl")
include("functions/adapt.jl")

export IsWrappedArray,
  is_wrapped_array, parenttype, unwrap_type, expose, Exposed, unexpose, cpu

end
