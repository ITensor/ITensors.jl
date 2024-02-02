using ArrayLayouts: ArrayLayouts

function ArrayLayouts.MemoryLayout(arraytype::Type{<:SparseArrayLike})
  return SparseLayout()
end
