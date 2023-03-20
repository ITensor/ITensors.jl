module NDTensorMetal

using NDTensors

if isdefined(Base, :get_extension)
  using Metal
else
  using ..Metal
end

include("imports.jl")

include("set_types.jl")

end