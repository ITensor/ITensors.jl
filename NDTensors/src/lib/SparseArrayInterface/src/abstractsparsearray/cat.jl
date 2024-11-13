# TODO: Change to `AnyAbstractSparseArray`.
function Base.cat(as::SparseArrayLike...; dims)
  return sparse_cat(as...; dims)
end
