using BlockArrays: AbstractBlockArray, BlocksView
using ..SparseArrayInterface: SparseArrayInterface, stored_length

function SparseArrayInterface.stored_length(a::AbstractBlockArray)
  return sum(b -> stored_length(b), blocks(a); init=zero(Int))
end

# TODO: Handle `BlocksView` wrapping a sparse array?
function SparseArrayInterface.storage_indices(a::BlocksView)
  return CartesianIndices(a)
end
