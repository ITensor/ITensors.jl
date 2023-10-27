# This was calling generic matrix multiplication.
# TODO: Raise an issue with `Metal.jl`.
function LinearAlgebra.mul!(
  CM::Exposed{<:MtlArray,<:Transpose},
  AM::Exposed{<:MtlArray},
  BM::Exposed{<:MtlArray},
  α,
  β,
)
  mul!(transpose(unexpose(CM)), transpose(unexpose(BM)), transpose(unexpose(AM)), α, β)
  return unexpose(CM)
end
