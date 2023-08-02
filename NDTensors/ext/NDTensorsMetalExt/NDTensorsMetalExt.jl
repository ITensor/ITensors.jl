module NDTensorsMetalExt

using NDTensors
using NDTensors.SetParameters
using Functors
using Adapt

if isdefined(Base, :get_extension)
  using Metal
else
  using ..Metal
end

include("imports.jl")
include("adapt.jl")
include("set_types.jl")
end
