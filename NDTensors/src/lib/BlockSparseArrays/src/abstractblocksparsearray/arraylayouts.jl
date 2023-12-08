using ArrayLayouts: ArrayLayouts, MemoryLayout, MatMulMatAdd, MulAdd
using BlockArrays: BlockLayout
using ..SparseArrayInterface: SparseLayout
using LinearAlgebra: mul!

# TODO: Generalize to `BlockSparseArrayLike`.
function ArrayLayouts.MemoryLayout(arraytype::Type{<:AbstractBlockSparseArray})
  outer_layout = typeof(MemoryLayout(blockstype(arraytype)))
  inner_layout = typeof(MemoryLayout(blocktype(arraytype)))
  return BlockLayout{outer_layout,inner_layout}()
end

function Base.similar(
  ::MulAdd{<:BlockLayout{<:SparseLayout},<:BlockLayout{<:SparseLayout}}, elt::Type, axes
)
  return similar(BlockSparseArray{elt}, axes)
end

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
