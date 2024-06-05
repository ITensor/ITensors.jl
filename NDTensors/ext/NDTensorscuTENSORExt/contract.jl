using NDTensors: NDTensors, DenseTensor, array, data
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
  ## TODO for now a hack to get cuTENSOR working with blocksparse
  R = unexpose(R)
  cuR = CuTensor(copy(array(R)), collect(labelsR))
  cuT1 = CuTensor(copy(array(unexpose(T1))), collect(labelsT1))
  cuT2 = CuTensor(copy(array(unexpose(T2))), collect(labelsT2))
  cuTENSOR.mul!(cuR, cuT1, cuT2, α, β)
  copyto!(data(R), cuR.data)
  return R
end
