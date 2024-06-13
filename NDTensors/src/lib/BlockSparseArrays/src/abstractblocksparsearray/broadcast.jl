using BlockArrays: AbstractBlockedUnitRange, BlockSlice
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
      <:Tuple{BlockSlice{<:Any,<:AbstractBlockedUnitRange},Vararg{Any}},
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
        BlockSlice{<:Any,<:AbstractBlockedUnitRange},
        BlockSlice{<:Any,<:AbstractBlockedUnitRange},
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
      <:Tuple{Any,BlockSlice{<:Any,<:AbstractBlockedUnitRange},Vararg{Any}},
    },
  },
)
  return BlockSparseArrayStyle{ndims(arraytype)}()
end
