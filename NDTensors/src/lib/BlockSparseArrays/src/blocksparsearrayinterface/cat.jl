using BlockArrays: AbstractBlockedUnitRange, blockedrange, blocklengths
using NDTensors.SparseArraysBase: SparseArraysBase, allocate_cat_output, sparse_cat!

# TODO: Maybe move to `SparseArraysBaseBlockArraysExt`.
# TODO: Handle dual graded unit ranges, for example in a new `SparseArraysBaseGradedAxesExt`.
function SparseArraysBase.axis_cat(
  a1::AbstractBlockedUnitRange, a2::AbstractBlockedUnitRange
)
  return blockedrange(vcat(blocklengths(a1), blocklengths(a2)))
end

# that erroneously allocates too many blocks that are
# zero and shouldn't be stored.
function blocksparse_cat!(a_dest::AbstractArray, as::AbstractArray...; dims)
  sparse_cat!(blocks(a_dest), blocks.(as)...; dims)
  return a_dest
end

# TODO: Delete this in favor of `sparse_cat`, currently
# that erroneously allocates too many blocks that are
# zero and shouldn't be stored.
function blocksparse_cat(as::AbstractArray...; dims)
  a_dest = allocate_cat_output(as...; dims)
  blocksparse_cat!(a_dest, as...; dims)
  return a_dest
end
