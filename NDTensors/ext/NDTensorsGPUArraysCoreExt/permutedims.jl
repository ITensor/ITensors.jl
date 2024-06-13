using NDTensors: NDTensors, data
using GPUArraysCore: @allowscalar, AbstractGPUArray
function NDTensors.permutedims!(
  Rexposed::Exposed{<:AbstractGPUArray,<:DiagTensor},
  texposed::Exposed{<:AbstractGPUArray,<:DiagTensor},
  perm::NTuple{N,Int},
  f::Function=(r, t) -> t,
) where {N}
  R = unexpose(Rexposed)
  t = unexpose(texposed)

  data(R) .= f.(data(R), data(t))
  return R
end
