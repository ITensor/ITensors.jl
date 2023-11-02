module NDTensorsMetalExt

using Adapt
using Functors
using LinearAlgebra: LinearAlgebra, Transpose, mul!
using NDTensors
using NDTensors.SetParameters

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
