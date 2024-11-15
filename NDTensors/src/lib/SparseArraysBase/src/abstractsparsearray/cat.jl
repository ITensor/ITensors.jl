# TODO: Change to `AnyAbstractSparseArray`.
function Base.cat(as::AnyAbstractSparseArray...; dims)
  return sparse_cat(as...; dims)
end
