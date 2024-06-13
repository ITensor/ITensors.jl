using NDTensors: NDTensors, array
using GPUArraysCore: @allowscalar, AbstractGPUArray
using NDTensors.Adapt
function NDTensors.permutedims!(
  Rexposed::Exposed{<:AbstractGPUArray, <:DiagTensor},
  texposed::Exposed{<:AbstractGPUArray, <:DiagTensor},
  perm::NTuple{N,Int},
  f::Function=(r, t) -> t,
) where {N}
  R = unexpose(Rexposed)
  t = unexpose(texposed)

  array(R) .= f.(array(R), array(t))
  return R
end