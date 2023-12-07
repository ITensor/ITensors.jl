using Base.Broadcast: Broadcast

function Broadcast.BroadcastStyle(arraytype::Type{<:BlockSparseArrayLike})
  return BlockSparseArrayStyle{ndims(arraytype)}()
end
