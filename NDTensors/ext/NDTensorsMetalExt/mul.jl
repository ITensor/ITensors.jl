# This was calling generic matrix multiplication.
# TODO: Raise an issue with `Metal.jl`.
function NDTensors.mul!!(
  ::Type{<:MtlArray},
  CM::Transpose,
  ::Type{<:MtlArray},
  AM::AbstractMatrix,
  ::Type{<:MtlArray},
  BM::AbstractMatrix,
  α,
  β,
)
  mul!(transpose(CM), transpose(BM), transpose(AM), α, β)
  return CM
end
