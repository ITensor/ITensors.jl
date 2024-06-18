using BlockArrays: BlockIndexRange, BlockRange, BlockSlice, block

# TODO: Define `AnyBlockSparseVector`.
function Base.view(a::BlockSparseArrayLike{<:Any,N}, index::Block{N}) where {N}
  return blocksparse_view(a, index)
end

# Fix ambiguity error with `BlockArrays`.
function Base.view(
  a::SubArray{
    <:Any,
    N,
    <:AbstractBlockSparseArray,
    <:Tuple{
      Vararg{
        Union{Base.Slice,BlockSlice{<:BlockRange{1,<:Tuple{AbstractUnitRange{Int}}}}},N
      },
    },
  },
  index::Block{N},
) where {N}
  return blocksparse_view(a, index)
end

# Fix ambiguity error with `BlockArrays`.
# TODO: Define `AnyBlockSparseVector`.
function Base.view(a::BlockSparseArrayLike{<:Any,1}, index::Block{1})
  return blocksparse_view(a, index)
end
