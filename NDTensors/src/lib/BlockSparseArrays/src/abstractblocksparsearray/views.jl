using BlockArrays: BlockArrays, Block, viewblock

function blocksparse_view(a, I...)
  return Base.invoke(view, Tuple{AbstractArray,Vararg{Any}}, a, I...)
end

# These definitions circumvent some generic definitions in BlockArrays.jl:
# https://github.com/JuliaArrays/BlockArrays.jl/blob/master/src/views.jl
# which don't handle subslices of blocks properly.
function Base.view(
  a::SubArray{
    <:Any,N,<:BlockSparseArrayLike,<:Tuple{Vararg{BlockSlice{<:BlockRange{1}},N}}
  },
  I::Block{N},
) where {N}
  return blocksparse_view(a, I)
end
function Base.view(
  a::SubArray{
    <:Any,N,<:BlockSparseArrayLike,<:Tuple{Vararg{BlockSlice{<:BlockRange{1}},N}}
  },
  I::Vararg{Block{1},N},
) where {N}
  return blocksparse_view(a, I...)
end
function Base.view(
  V::SubArray{<:Any,1,<:BlockSparseArrayLike,<:Tuple{BlockSlice{<:BlockRange{1}}}},
  I::Block{1},
)
  return blocksparse_view(a, I)
end

# Specialized code for getting the view of a block.
function BlockArrays.viewblock(
  a::AbstractBlockSparseArray{<:Any,N}, block::Block{N}
) where {N}
  return viewblock(a, Tuple(block)...)
end

# TODO: Define `blocksparse_viewblock`.
function BlockArrays.viewblock(
  a::AbstractBlockSparseArray{<:Any,N}, block::Vararg{Block{1},N}
) where {N}
  I = CartesianIndex(Int.(block))
  if I ∈ stored_indices(blocks(a))
    return blocks(a)[I]
  end
  return BlockView(a, block)
end

function Base.view(
  a::SubArray{T,N,<:AbstractBlockSparseArray{T,N},<:Tuple{Vararg{BlockSliceCollection,N}}},
  block::Block{N},
) where {T,N}
  return viewblock(a, block)
end
function Base.view(
  a::SubArray{T,N,<:AbstractBlockSparseArray{T,N},<:Tuple{Vararg{BlockSliceCollection,N}}},
  block::Vararg{Block{1},N},
) where {T,N}
  return viewblock(a, block...)
end
function BlockArrays.viewblock(
  a::SubArray{T,N,<:AbstractBlockSparseArray{T,N},<:Tuple{Vararg{BlockSliceCollection,N}}},
  block::Block{N},
) where {T,N}
  return viewblock(a, Tuple(block)...)
end

# Fixes ambiguity error with `BlockSparseArrayLike` definition.
function Base.view(
  a::SubArray{
    T,N,<:AbstractBlockSparseArray{T,N},<:Tuple{Vararg{BlockSlice{<:BlockRange{1}},N}}
  },
  block::Block{N},
) where {T,N}
  return viewblock(a, block)
end
# Fixes ambiguity error with `BlockSparseArrayLike` definition.
function Base.view(
  a::SubArray{
    T,N,<:AbstractBlockSparseArray{T,N},<:Tuple{Vararg{BlockSlice{<:BlockRange{1}},N}}
  },
  block::Vararg{Block{1},N},
) where {T,N}
  return viewblock(a, block...)
end

# TODO: Define `blocksparse_viewblock`.
function BlockArrays.viewblock(
  a::SubArray{T,N,<:AbstractBlockSparseArray{T,N},<:Tuple{Vararg{BlockSliceCollection,N}}},
  block::Vararg{Block{1},N},
) where {T,N}
  I = CartesianIndex(Int.(block))
  if I ∈ stored_indices(blocks(a))
    return blocks(a)[I]
  end
  return BlockView(a, block)
end
