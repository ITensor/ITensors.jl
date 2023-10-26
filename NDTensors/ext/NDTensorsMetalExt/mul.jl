# This was calling generic matrix multiplication.
# TODO: Raise an issue with `Metal.jl`.
function LinearAlgebra.mul!(
  CM::Unwrap.Exposed{<:MtlArray},
  AM::Unwrap.Exposed{<:MtlArray},
  BM::Unwrap.Exposed{<:MtlArray},
  α,
  β,
)
  return mul!(transpose(CM.object), transpose(BM.object), transpose(AM.object), α, β)
end
