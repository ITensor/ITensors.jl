module Unwrap
using SimpleTraits
using LinearAlgebra
using Base: ReshapedArray
using Strided.StridedViews

include("expose.jl")
include("iswrappedarray.jl")

include("import.jl")
## TODO Create functions which take the `Expose` type and launch functions
## using that type
## Exposed based functions
include("functions/util.jl")
include("functions/copyto.jl")
include("functions/linearalgebra.jl")
include("functions/mul.jl")
include("functions/permutedims.jl")

export IsWrappedArray,
  is_wrapped_array, parenttype, unwrap_type, expose, Exposed, unexpose, cpu

## TODO write exposed based functions in the NDTensors Extensions when necessary

end
