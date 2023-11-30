# Broadcasting
function Broadcast.BroadcastStyle(arraytype::Type{<:SparseArrayDOK})
  return SparseArrayInterface.SparseArrayStyle{ndims(arraytype)}()
end
