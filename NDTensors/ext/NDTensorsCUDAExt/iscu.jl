using CUDA: CuArray
using NDTensors: NDTensors

NDTensors.iscu(::Type{<:CuArray}) = true
