using BlockArrays: BlockIndexRange, block

function Base.view(a::BlockSparseArrayLike{<:Any,N}, index::Block{N}) where {N}
  return blocks(a)[Int.(Tuple(index))...]
end

# TODO: Define `AnyBlockSparseVector`.
function Base.view(a::BlockSparseArrayLike{<:Any,1}, index::Block{1})
  return blocks(a)[Int.(Tuple(index))...]
end

function Base.view(a::BlockSparseArrayLike, indices::BlockIndexRange)
  return view(view(a, block(indices)), indices.indices...)
end
