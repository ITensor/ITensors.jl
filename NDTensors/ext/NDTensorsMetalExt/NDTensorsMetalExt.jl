module NDTensorsMetalExt

using Adapt
using Functors
using LinearAlgebra: LinearAlgebra
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
include("permutedims.jl")
end
