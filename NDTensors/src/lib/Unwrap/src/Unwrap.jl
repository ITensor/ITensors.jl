module Unwrap
using Adapt: Adapt, adapt, adapt_structure

include("expose.jl")
include("unwraptype.jl")

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
  unwrap_type, expose, Exposed, unexpose, cpu

end
