using ArrayLayouts: ArrayLayouts, Dot, MatMulMatAdd, MatMulVecAdd, MulAdd
using BlockArrays: BlockLayout
using ..SparseArraysBase: SparseLayout
using LinearAlgebra: dot, mul!

function blocksparse_muladd!(
  α::Number, a1::AbstractArray, a2::AbstractArray, β::Number, a_dest::AbstractArray
)
  mul!(blocks(a_dest), blocks(a1), blocks(a2), α, β)
  return a_dest
end

function blocksparse_matmul!(m::MulAdd)
  α, a1, a2, β, a_dest = m.α, m.A, m.B, m.β, m.C
  blocksparse_muladd!(α, a1, a2, β, a_dest)
  return a_dest
end

function ArrayLayouts.materialize!(
  m::MatMulMatAdd{
    <:BlockLayout{<:SparseLayout},
    <:BlockLayout{<:SparseLayout},
    <:BlockLayout{<:SparseLayout},
  },
)
  blocksparse_matmul!(m)
  return m.C
end
function ArrayLayouts.materialize!(
  m::MatMulVecAdd{
    <:BlockLayout{<:SparseLayout},
    <:BlockLayout{<:SparseLayout},
    <:BlockLayout{<:SparseLayout},
  },
)
  blocksparse_matmul!(m)
  return m.C
end

function blocksparse_dot(a1::AbstractArray, a2::AbstractArray)
  # TODO: Add a check that the blocking of `a1` and `a2` are
  # the same, or the same up to a reshape.
  return dot(blocks(a1), blocks(a2))
end

function Base.copy(d::Dot{<:BlockLayout{<:SparseLayout},<:BlockLayout{<:SparseLayout}})
  return blocksparse_dot(d.A, d.B)
end
