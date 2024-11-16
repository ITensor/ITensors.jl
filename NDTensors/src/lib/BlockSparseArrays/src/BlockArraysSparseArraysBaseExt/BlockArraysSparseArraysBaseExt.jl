using BlockArrays: AbstractBlockArray, BlocksView
using ..SparseArraysBase: SparseArraysBase, stored_length

function SparseArraysBase.stored_length(a::AbstractBlockArray)
  return sum(b -> stored_length(b), blocks(a); init=zero(Int))
end

# TODO: Handle `BlocksView` wrapping a sparse array?
function SparseArraysBase.storage_indices(a::BlocksView)
  return CartesianIndices(a)
end
