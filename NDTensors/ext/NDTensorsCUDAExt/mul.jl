# This was calling generic matrix multiplication.
# TODO: Raise an issue with `CUDA.jl`.
function LinearAlgebra.mul!(
  CM::Exposed{<:CuArray,<:LinearAlgebra.Transpose},
  AM::Exposed{<:CuArray},
  BM::Exposed{<:CuArray},
  α,
  β,
)
  mul!(transpose(CM), transpose(BM), transpose(AM), α, β)
  return unexpose(CM)
end

# This was calling generic matrix multiplication.
# TODO: Raise an issue with `CUDA.jl`.
function LinearAlgebra.mul!(
  CM::Exposed{<:CuArray,<:LinearAlgebra.Adjoint},
  AM::Exposed{<:CuArray},
  BM::Exposed{<:CuArray},
  α,
  β,
)
  mul!(CM', BM', AM', α, β)
  return unexpose(CM)
end
