using NDTensors: Tensor
using Adapt: adapt
using GPUArraysCore: @allowscalar

function promote_tensor_eltype(
  ::Type{<:ElT1}, T1::Tensor{ElT1}, T2::Tensor{ElT1}
) where {ElT1}
  return T1, T2
end

function promote_tensor_eltype(
  ::Type{<:ElT1}, T1::Tensor{ElT1}, T2::Tensor{ElT2}
) where {ElT1,ElT2}
  T2 = @allowscalar adapt(ElT1, T2)
  return T1, T2
end

function promote_tensor_eltype(
  ::Type{<:ElT1}, T1::Tensor{ElT2}, T2::Tensor{ElT1}
) where {ElT1,ElT2}
  T1 = @allowscalar adapt(ElT1, T1)
  return T1, T2
end
