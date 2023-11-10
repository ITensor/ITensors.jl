# This was calling generic matrix multiplication.
# TODO: Raise an issue with `CUDA.jl`.
function mul!(
  CM::Exposed{<:CuArray,<:LinearAlgebra.Transpose},
  AM::Exposed{<:CuArray},
  BM::Exposed{<:CuArray},
  α,
  β,
)
  return mul!(parent(CM), transpose(BM), transpose(AM), α, β)

  return unexpose(CM)
end

# This was calling generic matrix multiplication.
# TODO: Raise an issue with `CUDA.jl`.
function mul!(
  CM::Exposed{<:CuArray,<:LinearAlgebra.Adjoint},
  AM::Exposed{<:CuArray},
  BM::Exposed{<:CuArray},
  α,
  β,
)
  return mul!(parent(CM), BM', AM', α, β)

  return unexpose(CM)
end
