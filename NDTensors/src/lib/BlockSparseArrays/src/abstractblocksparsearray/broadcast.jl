using BlockArrays: BlockedUnitRange, BlockSlice
using Base.Broadcast: Broadcast

function Broadcast.BroadcastStyle(arraytype::Type{<:BlockSparseArrayLike})
  return BlockSparseArrayStyle{ndims(arraytype)}()
end

# Fix ambiguity error with `BlockArrays`.
function Broadcast.BroadcastStyle(
  arraytype::Type{
    <:SubArray{
      <:Any,
      <:Any,
      <:AbstractBlockSparseArray,
      <:Tuple{BlockSlice{<:Any,<:BlockedUnitRange},Vararg{Any}},
    },
  },
)
  return BlockSparseArrayStyle{ndims(arraytype)}()
end
function Broadcast.BroadcastStyle(
  arraytype::Type{
    <:SubArray{
      <:Any,
      <:Any,
      <:AbstractBlockSparseArray,
      <:Tuple{
        BlockSlice{<:Any,<:BlockedUnitRange},
        BlockSlice{<:Any,<:BlockedUnitRange},
        Vararg{Any},
      },
    },
  },
)
  return BlockSparseArrayStyle{ndims(arraytype)}()
end
function Broadcast.BroadcastStyle(
  arraytype::Type{
    <:SubArray{
      <:Any,
      <:Any,
      <:AbstractBlockSparseArray,
      <:Tuple{Any,BlockSlice{<:Any,<:BlockedUnitRange},Vararg{Any}},
    },
  },
)
  return BlockSparseArrayStyle{ndims(arraytype)}()
end
