using BlockArrays: BlockArrays, Block, BlockSlices, viewblock

function blocksparse_view(a, I...)
  return Base.invoke(view, Tuple{AbstractArray,Vararg{Any}}, a, I...)
end

# These definitions circumvent some generic definitions in BlockArrays.jl:
# https://github.com/JuliaArrays/BlockArrays.jl/blob/master/src/views.jl
# which don't handle subslices of blocks properly.
function Base.view(
  a::SubArray{<:Any,N,<:BlockSparseArrayLike,<:NTuple{N,BlockSlices}}, I::Block{N}
) where {N}
  return blocksparse_view(a, I)
end
function Base.view(
  a::SubArray{<:Any,N,<:BlockSparseArrayLike,<:NTuple{N,BlockSlices}}, I::Vararg{Block{1},N}
) where {N}
  return blocksparse_view(a, I...)
end
function Base.view(
  V::SubArray{<:Any,1,<:BlockSparseArrayLike,<:Tuple{BlockSlices}}, I::Block{1}
)
  return blocksparse_view(a, I)
end

# Specialized code for getting the view of a block.
function BlockArrays.viewblock(
  a::AbstractBlockSparseArray{<:Any,N}, block::Block{N}
) where {N}
  return viewblock(a, Tuple(block)...)
end
function BlockArrays.viewblock(
  a::AbstractBlockSparseArray{<:Any,N}, block::Vararg{Block{1},N}
) where {N}
  I = CartesianIndex(Int.(block))
  if I ∈ stored_indices(blocks(a))
    return blocks(a)[I]
  end
  return BlockView(a, block)
end
