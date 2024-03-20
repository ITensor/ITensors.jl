module NDTensorsAMDGPUExt
using Functors

include("copyto.jl")
include("set_types.jl")
include("adapt.jl")
include("indexing.jl")
include("linearalgebra.jl")
include("mul.jl")
include("permutedims.jl")

end
