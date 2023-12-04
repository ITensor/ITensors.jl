using LinearAlgebra: LinearAlgebra, mul!

function LinearAlgebra.mul!(
  a_dest::AbstractMatrix,
  a1::AbstractBlockSparseMatrix,
  a2::AbstractBlockSparseMatrix,
  α::Number=true,
  β::Number=false,
)
  mul!(blocks(a_dest), blocks(a1), blocks(a2), α, β)
  return a_dest
end
