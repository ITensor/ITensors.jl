using ArrayLayouts: ArrayLayouts, Dot, DualLayout, MatMulMatAdd, MatMulVecAdd, MulAdd
using LinearAlgebra: Adjoint, Transpose
using ..TypeParameterAccessors: parenttype

function ArrayLayouts.MemoryLayout(arraytype::Type{<:AnyAbstractSparseArray})
  return SparseLayout()
end

# TODO: Generalize to `SparseVectorLike`/`AnySparseVector`.
function ArrayLayouts.MemoryLayout(arraytype::Type{<:Adjoint{<:Any,<:AbstractSparseVector}})
  return DualLayout{typeof(MemoryLayout(parenttype(arraytype)))}()
end
# TODO: Generalize to `SparseVectorLike`/`AnySparseVector`.
function ArrayLayouts.MemoryLayout(
  arraytype::Type{<:Transpose{<:Any,<:AbstractSparseVector}}
)
  return DualLayout{typeof(MemoryLayout(parenttype(arraytype)))}()
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
