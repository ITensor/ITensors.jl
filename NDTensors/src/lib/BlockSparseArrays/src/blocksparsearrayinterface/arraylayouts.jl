using ArrayLayouts: ArrayLayouts, MatMulMatAdd
using BlockArrays: BlockLayout
using ..SparseArrayInterface: SparseLayout
using LinearAlgebra: mul!

function blocksparse_muladd!(
  α::Number, a1::AbstractMatrix, a2::AbstractMatrix, β::Number, a_dest::AbstractMatrix
)
  mul!(blocks(a_dest), blocks(a1), blocks(a2), α, β)
  return a_dest
end

function ArrayLayouts.materialize!(
  m::MatMulMatAdd{
    <:BlockLayout{<:SparseLayout},
    <:BlockLayout{<:SparseLayout},
    <:BlockLayout{<:SparseLayout},
  },
)
  α, a1, a2, β, a_dest = m.α, m.A, m.B, m.β, m.C
  blocksparse_muladd!(α, a1, a2, β, a_dest)
  return a_dest
end
