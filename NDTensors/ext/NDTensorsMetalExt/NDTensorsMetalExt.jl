module NDTensorsMetalExt

using Adapt
using Functors
using LinearAlgebra: LinearAlgebra, Adjoint, Transpose, mul!, qr, eigen, svd
using NDTensors
using NDTensors.SetParameters
using NDTensors.Unwrap: qr_positive, ql_positive, ql

using Metal

include("imports.jl")
include("adapt.jl")
include("set_types.jl")
include("indexing.jl")
include("linearalgebra.jl")
include("copyto.jl")
include("append.jl")
include("permutedims.jl")
include("mul.jl")
include("splice.jl")

end
