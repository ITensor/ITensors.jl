using Base.Broadcast: Broadcast

function Broadcast.BroadcastStyle(arraytype::Type{<:AbstractBlockSparseArray})
  return BlockSparseArrayStyle{ndims(arraytype)}()
end

function Broadcast.BroadcastStyle(arraytype::Type{<:PermutedDimsBlockSparseArray})
  return BlockSparseArrayStyle{ndims(arraytype)}()
end
