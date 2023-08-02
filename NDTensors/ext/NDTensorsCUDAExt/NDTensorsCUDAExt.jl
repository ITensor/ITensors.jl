module NDTensorsCUDAExt

using NDTensors
using NDTensors.SetParameters
using Adapt
using Functors
using LinearAlgebra: BlasFloat

if isdefined(Base, :get_extension)
  using CUDA
  using CUDA.CUBLAS
  using CUDA.CUSOLVER
else
  using ..CUDA
  using .CUBLAS
  using .CUSOLVER
end

include("imports.jl")
include("set_types.jl")
include("adapt.jl")
include("linearalgebra.jl")
end
