module NDTensorCUDA

using NDTensors
if isdefined(Base, :get_extension)
  using CUDA
else
  using ..CUDA
end

include("imports.jl")

include("set_types.jl")

end
