using ArrayLayouts: ArrayLayouts, Dot, MatMulMatAdd, MatMulVecAdd, MulAdd

function ArrayLayouts.MemoryLayout(arraytype::Type{<:SparseArrayLike})
  return SparseLayout()
end

function sparse_matmul!(m::MulAdd)
  α, a1, a2, β, a_dest = m.α, m.A, m.B, m.β, m.C
  sparse_mul!(a_dest, a1, a2, α, β)
  return a_dest
end

function ArrayLayouts.materialize!(
  m::MatMulMatAdd{<:AbstractSparseLayout,<:AbstractSparseLayout,<:AbstractSparseLayout}
)
  sparse_matmul!(m)
  return m.C
end
function ArrayLayouts.materialize!(
  m::MatMulVecAdd{<:AbstractSparseLayout,<:AbstractSparseLayout,<:AbstractSparseLayout}
)
  sparse_matmul!(m)
  return m.C
end

function Base.copy(d::Dot{<:SparseLayout,<:SparseLayout})
  return sparse_dot(d.A, d.B)
end
