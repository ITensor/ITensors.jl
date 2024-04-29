using NDTensors:
  NDTensors,
  DenseTensor,
  Tensor,
  array,
  contract!,
  contraction_output,
  contraction_output_type,
  contract_inds,
  dense,
  inds
using NDTensors.Expose: Exposed, expose, unexpose
using cuTENSOR: CuArray, CuTensor, mul!

function NDTensors.contract!(
  R::Exposed{<:CuArray,<:DenseTensor{ElT}},
  labelsR,
  T1::Exposed{<:CuArray,<:DenseTensor},
  labelsT1,
  T2::Exposed{<:CuArray,<:DenseTensor},
  labelsT2,
  α::Elα=one(Bool),
  β::Elβ=zero(Bool),
) where {Elα<:Number,Elβ<:Number,ElT}
  cuR, cuT1, cuT2 =
    CuTensor.(
      array.((unexpose(R), unexpose(T1), unexpose(T2))),
      collect.((labelsR, labelsT1, labelsT2)),
    )
  cuTENSOR.mul!(cuR, cuT1, cuT2, α, β)
  return output
end
