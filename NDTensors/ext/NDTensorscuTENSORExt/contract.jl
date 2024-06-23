using NDTensors: NDTensors, DenseTensor, array
using NDTensors.Expose: Exposed, unexpose
using cuTENSOR: cuTENSOR, CuArray, CuTensor

# Handle cases that can't be handled by `cuTENSOR.jl`
# right now.
function to_zero_offset_cuarray(a::CuArray)
  return iszero(a.offset) ? a : copy(a)
end
function to_zero_offset_cuarray(a::ReshapedArray)
  return copy(a)
end

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
  arrayT1 = to_zero_offset_cuarray(array(T1))
  arrayT2 = to_zero_offset_cuarray(array(T2))
  # Promote to a common type.
  elt = promote_type(eltype(arrayT1), eltype(arrayT2))
  arrayT1 = convert(CuArray{elt}, arrayT1)
  arrayT2 = convert(CuArray{elt}, arrayT2)
  cuR = CuTensor(arrayR, collect(labelsR))
  cuT1 = CuTensor(arrayT1, collect(labelsT1))
  cuT2 = CuTensor(arrayT2, collect(labelsT2))
  cuTENSOR.mul!(cuR, cuT1, cuT2, α, β)
  if !zoffR
    array(R) .= cuR.data
  end
  return R
end
