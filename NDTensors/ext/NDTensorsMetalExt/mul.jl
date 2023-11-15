# This was calling generic matrix multiplication.
# TODO: Raise an issue with `Metal.jl`.
function LinearAlgebra.mul!(
  CM::Exposed{<:MtlArray,<:Transpose},
  AM::Exposed{<:MtlArray},
  BM::Exposed{<:MtlArray},
  α,
  β,
)
  mul!(transpose(CM), transpose(BM), transpose(AM), α, β)
  return unexpose(CM)
end

# This was calling generic matrix multiplication.
# TODO: Raise an issue with `Metal.jl`.
function LinearAlgebra.mul!(
  CM::Exposed{<:MtlArray,<:Adjoint}, AM::Exposed{<:MtlArray}, BM::Exposed{<:MtlArray}, α, β
)
  mul!(CM', BM', AM', α, β)
  return unexpose(CM)
end
