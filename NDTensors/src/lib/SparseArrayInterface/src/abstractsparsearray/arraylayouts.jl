using ArrayLayouts: ArrayLayouts, MemoryLayout, MulAdd

abstract type AbstractSparseLayout <: MemoryLayout end

struct SparseLayout <: AbstractSparseLayout end

function ArrayLayouts.MemoryLayout(arraytype::Type{<:AbstractSparseArray})
  return SparseLayout()
end
