module NDTensorsMetalExt

using Adapt
using Functors
using LinearAlgebra: LinearAlgebra, Adjoint, Transpose, mul!, qr, eigen, svd
using NDTensors
using NDTensors.Unwrap: qr_positive, ql_positive, ql

using Metal

include("typeparameteraccessors.jl")
include("imports.jl")
include("adapt.jl")
include("indexing.jl")
include("linearalgebra.jl")
include("copyto.jl")
include("append.jl")
include("permutedims.jl")
include("mul.jl")

end
