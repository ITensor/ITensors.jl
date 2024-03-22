using ArrayLayouts: ArrayLayouts, MatMulMatAdd
using BlockArrays: BlockLayout
using ..SparseArrayInterface: SparseLayout
using LinearAlgebra: mul!

function ArrayLayouts.materialize!(
  m::MatMulMatAdd{
    <:BlockLayout{<:SparseLayout},
    <:BlockLayout{<:SparseLayout},
    <:BlockLayout{<:SparseLayout},
  },
)
  α, a1, a2, β, a_dest = m.α, m.A, m.B, m.β, m.C
  mul!(a_dest, a1, a2, α, β)
  return a_dest
end
