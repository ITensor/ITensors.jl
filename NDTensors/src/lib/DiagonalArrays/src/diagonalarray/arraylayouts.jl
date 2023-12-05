using ArrayLayouts: MulAdd

# Default sparse array type for `AbstractDiagonalLayout`.
default_diagonalarraytype(elt::Type) = DiagonalArray{elt}

# TODO: Preserve GPU memory! Implement `CuSparseArrayLayout`, `MtlSparseLayout`?
function Base.similar(::MulAdd{<:AbstractDiagonalLayout,<:AbstractDiagonalLayout}, elt::Type, axes)
  return similar(default_diagonalarraytype(elt), axes)
end
