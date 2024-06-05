using NDTensors: NDTensors, DenseTensor, array, data
using NDTensors.Expose: Exposed, unexpose
using cuTENSOR: cuTENSOR, CuArray, CuTensor

struct CuArrayOffset{N} end
function CuArrayOffset(V)
  return CuArrayOffset{V}()
end

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
  R, T1, T2 = unexpose.((R, T1, T2))
  ## Instance the contract! function off the 
  ## offset on the CuArrays
  offR = CuArrayOffset(data(R).offset)
  offT1 = CuArrayOffset(data(T1).offset)
  offT2 = CuArrayOffset(data(T2).offset)
  return contract!(offR, R, labelsR, offT1, T1, labelsT1, offT2, T2, labelsT2, α, β)
end

function contract!(
  ::CuArrayOffset{0},
  R,
  labelsR,
  ::CuArrayOffset{0},
  T1,
  labelsT1,
  ::CuArrayOffset{0},
  T2,
  labelsT2,
  α,
  β,
)
  cuR = CuTensor(array(R), collect(labelsR))
  cuT1 = CuTensor(array(T1), collect(labelsT1))
  cuT2 = CuTensor(array(T2), collect(labelsT2))
  cuTENSOR.mul!(cuR, cuT1, cuT2, α, β)
  return R
end

## TODO Should I always copy all of them if any are non-zero
## Or should I create a function for every possible non-zero case?
function contract!(
  ::CuArrayOffset,
  R,
  labelsR,
  ::CuArrayOffset,
  T1,
  labelsT1,
  ::CuArrayOffset,
  T2,
  labelsT2,
  α,
  β,
)
  cuR = CuTensor(copy(array(R)), collect(labelsR))
  cuT1 = CuTensor(copy(array(T1)), collect(labelsT1))
  cuT2 = CuTensor(copy(array(T2)), collect(labelsT2))
  cuTENSOR.mul!(cuR, cuT1, cuT2, α, β)
  array(R) .= cuR.data
  return R
end
