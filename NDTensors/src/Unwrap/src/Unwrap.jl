module Unwrap
using SimpleTraits
using LinearAlgebra
using Base: ReshapedArray
using Strided.StridedViews

include("iswrappedarray.jl")
include("expose.jl")

include("import.jl")
## Exposed based functions
include("functions/permutedims.jl")

export IsWrappedArray, is_wrapped_array, parenttype, unwrap_type, expose, Expose

## TODO Create functions which take the `Expose` type and launch functions
## using that type
## TODO write exposed based functions in the NDTensors Extensions when necessary

end
