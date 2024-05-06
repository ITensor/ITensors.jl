using ..BackendSelection: @Algorithm_str
using LinearAlgebra: mul!

function contract!(
  alg::Algorithm"matricize",
  a_dest::AbstractArray,
  biperm_dest::BlockedPermutation,
  a1::AbstractArray,
  biperm1::BlockedPermutation,
  a2::AbstractArray,
  biperm2::BlockedPermutation,
  α::Number,
  β::Number,
)
  a_dest_mat = fusedims(a_dest, biperm_dest)
  a1_mat = fusedims(a1, biperm1)
  a2_mat = fusedims(a2, biperm2)
  _mul!(a_dest_mat, a1_mat, a2_mat, α, β)
  splitdims!(a_dest, a_dest_mat, biperm_dest)
  return a_dest
end

# Matrix multiplication.
function _mul!(
  a_dest::AbstractMatrix, a1::AbstractMatrix, a2::AbstractMatrix, α::Number, β::Number
)
  mul!(a_dest, a1, a2, α, β)
  return a_dest
end

# Inner product.
function _mul!(
  a_dest::AbstractArray{<:Any,0},
  a1::AbstractVector,
  a2::AbstractVector,
  α::Number,
  β::Number,
)
  a_dest[] = transpose(a1) * a2 * α + a_dest[] * β
  return a_dest
end
