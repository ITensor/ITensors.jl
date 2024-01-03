using LinearAlgebra: LinearAlgebra

LinearAlgebra.norm(a::AbstractSparseArray, p::Real=2) = sparse_norm(a, p)

# a1 * a2 * α + a_dest * β
function LinearAlgebra.mul!(
  a_dest::AbstractMatrix,
  a1::AbstractSparseMatrix,
  a2::AbstractSparseMatrix,
  α::Number=true,
  β::Number=false,
)
  sparse_mul!(a_dest, a1, a2, α, β)
  return a_dest
end
