module NDTensorsMetalExt

using Adapt
using Functors
using LinearAlgebra: LinearAlgebra, Transpose, mul!, qr, eigen, svd
using NDTensors
using NDTensors.SetParameters
using NDTensors.Unwrap: qr_positive, ql_positive, ql

if isdefined(Base, :get_extension)
  using Metal
else
  using ..Metal
end

include("imports.jl")
include("adapt.jl")
include("set_types.jl")
include("indexing.jl")
include("linearalgebra.jl")
include("copyto.jl")
include("append.jl")
include("permutedims.jl")
include("mul.jl")
end
