using NDTensors:
  NDTensors,
  DenseTensor,
  array
using NDTensors.Expose: Exposed, unexpose
using cuTENSOR: CuArray, CuTensor, mul!

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
  cuR, cuT1, cuT2 =
    CuTensor.(
      array.((unexpose(R), unexpose(T1), unexpose(T2))),
      collect.((labelsR, labelsT1, labelsT2)),
    )
  mul!(cuR, cuT1, cuT2, α, β)
  return R
end
