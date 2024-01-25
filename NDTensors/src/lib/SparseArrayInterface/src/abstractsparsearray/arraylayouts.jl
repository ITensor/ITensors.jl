using ArrayLayouts: ArrayLayouts, MemoryLayout, MulAdd

abstract type AbstractSparseLayout <: MemoryLayout end

struct SparseLayout <: AbstractSparseLayout end

function ArrayLayouts.MemoryLayout(arraytype::Type{<:SparseArrayLike})
  return SparseLayout()
end

function ArrayLayouts.sub_materialize(layout::AbstractSparseLayout, a, axes)
  error()
end
