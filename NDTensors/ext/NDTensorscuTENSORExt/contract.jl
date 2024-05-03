using NDTensors: NDTensors, DenseTensor, array
using NDTensors.Expose: Exposed, unexpose
using cuTENSOR: cuTENSOR, CuArray, CuTensor

function NDTensors.contract!(
  R::Exposed{<:CuArray,<:DenseTensor},
  labelsR,
  T1::Exposed{<:CuArray,<:DenseTensor},
  labelsT1,
  T2::Exposed{<:CuArray,<:DenseTensor},
  labelsT2,
  α::Number=one(Bool),
  β::Number=zero(Bool),
)
  cuR = CuTensor(array(unexpose(R)), collect(labelsR))
  cuT1 = CuTensor(array(unexpose(T1)), collect(labelsT1))
  cuT2 = CuTensor(array(unexpose(T2)), collect(labelsT2))
  cuTENSOR.mul!(cuR, cuT1, cuT2, α, β)
  return R
end
