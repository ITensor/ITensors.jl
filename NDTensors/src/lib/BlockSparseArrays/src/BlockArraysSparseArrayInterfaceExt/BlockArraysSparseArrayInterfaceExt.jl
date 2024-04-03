using BlockArrays: AbstractBlockArray, BlocksView
using ..SparseArrayInterface: SparseArrayInterface, nstored

function SparseArrayInterface.nstored(a::AbstractBlockArray)
  return sum(b -> nstored(b), blocks(a); init=zero(Int))
end

# TODO: Handle `BlocksView` wrapping a sparse array?
function SparseArrayInterface.storage_indices(a::BlocksView)
  return CartesianIndices(a)
end
