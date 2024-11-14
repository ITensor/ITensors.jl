# TODO: Change to `AnyAbstractBlockSparseArray`.
function Base.cat(as::AnyAbstractBlockSparseArray...; dims)
  # TODO: Use `sparse_cat` instead, currently
  # that erroneously allocates too many blocks that are
  # zero and shouldn't be stored.
  return blocksparse_cat(as...; dims)
end
