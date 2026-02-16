module Expose

using Adapt: Adapt, adapt, adapt_structure
using Base: ReshapedArray
using LinearAlgebra
using SimpleTraits
using StridedViews

include("exposed.jl")

include("import.jl")
## TODO Create functions which take the `Expose` type and launch functions
## using that type
## Exposed based functions
include("functions/abstractarray.jl")
include("functions/append.jl")
include("functions/copyto.jl")
include("functions/linearalgebra.jl")
include("functions/mul.jl")
include("functions/permutedims.jl")
include("functions/adapt.jl")

export IsWrappedArray, expose, Exposed, unexpose, cpu

end
