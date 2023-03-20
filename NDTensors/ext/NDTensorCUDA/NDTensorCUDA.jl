module NDTensorCUDA

using NDTensors
using Adapt
using Functors

if isdefined(Base, :get_extension)
  using CUDA
  using CUDA.CUBLAS
  using CUDA.CUSOLVER
else
  using ..CUDA
  using ..CUDA.CUBLAS
  using ..CUDA.CUSOLVER
end

include("imports.jl")

include("set_types.jl")
include("adapt.jl")
end
