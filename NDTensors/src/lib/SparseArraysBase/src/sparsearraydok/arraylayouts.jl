using ArrayLayouts: ArrayLayouts, MemoryLayout, MulAdd

ArrayLayouts.MemoryLayout(::Type{<:SparseArrayDOK}) = SparseLayout()

# Default sparse array type for `AbstractSparseLayout`.
default_sparsearraytype(elt::Type) = SparseArrayDOK{elt}

# TODO: Preserve GPU memory! Implement `CuSparseArrayLayout`, `MtlSparseLayout`?
function Base.similar(
  ::MulAdd{<:AbstractSparseLayout,<:AbstractSparseLayout}, elt::Type, axes
)
  return similar(default_sparsearraytype(elt), axes)
end
