using ..SparseArrayInterface: SparseArrayInterface, nstored

function SparseArrayInterface.nstored(a::AbstractBlockArray)
  return sum(b -> nstored(b), blocks(a); init=zero(Int))
end
