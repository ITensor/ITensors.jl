using NDTensors: NDTensors, DenseTensor, array, data
using NDTensors.Expose: Exposed, unexpose
using cuTENSOR: cuTENSOR, CuArray, CuTensor

function NDTensors.contract!(
  exposedR::Exposed{<:CuArray,<:DenseTensor},
  labelsR,
  exposedT1::Exposed{<:CuArray,<:DenseTensor},
  labelsT1,
  exposedT2::Exposed{<:CuArray,<:DenseTensor},
  labelsT2,
  α::Number=one(Bool),
  β::Number=zero(Bool),
)
  R, T1, T2 = unexpose.((exposedR, exposedT1, exposedT2))
  zoffR = iszero(array(R).offset)
  arrayR = zoffR ? array(R) : copy(array(R))
  arrayT1 = iszero(array(T1).offset) ? array(T1) : copy(array(T1))
  arrayT2 = iszero(array(T2).offset) ? array(T2) : copy(array(T2))
  cuR = CuTensor(arrayR, collect(labelsR))
  cuT1 = CuTensor(arrayT1, collect(labelsT1))
  cuT2 = CuTensor(arrayT2, collect(labelsT2))
  cuTENSOR.mul!(cuR, cuT1, cuT2, α, β)
  if !zoffR
    ## use vec to flatten cuR.data which could be multidimensional but
    ## tensor data is currently a vector
    data(R) .= vec(cuR.data)
  end
  return R
end
