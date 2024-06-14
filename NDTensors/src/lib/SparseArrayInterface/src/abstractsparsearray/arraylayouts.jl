using ArrayLayouts: ArrayLayouts, MatMulMatAdd

function ArrayLayouts.MemoryLayout(arraytype::Type{<:SparseArrayLike})
  return SparseLayout()
end

function ArrayLayouts.materialize!(
  m::MatMulMatAdd{<:AbstractSparseLayout,<:AbstractSparseLayout,<:AbstractSparseLayout}
)
  α, a1, a2, β, a_dest = m.α, m.A, m.B, m.β, m.C
  sparse_mul!(a_dest, a1, a2, α, β)
  return a_dest
end
