using BlockArrays: BlockArrays, Block, BlockedUnitRange, blockedrange, blocklength
using Dictionaries: Dictionary
using ..SparseArrayDOKs: SparseArrayDOK

# TODO: Delete this.
## using BlockArrays: blocks

struct BlockSparseArray{
  T,
  N,
  A<:AbstractArray{T,N},
  Blocks<:AbstractArray{A,N},
  Axes<:Tuple{Vararg{AbstractUnitRange,N}},
} <: AbstractBlockSparseArray{T,N}
  blocks::Blocks
  axes::Axes
end

const BlockSparseMatrix{T,A,Blocks,Axes} = BlockSparseArray{T,2,A,Blocks,Axes}
const BlockSparseVector{T,A,Blocks,Axes} = BlockSparseArray{T,1,A,Blocks,Axes}

function BlockSparseArray(
  block_data::Dictionary{<:Block{N},<:AbstractArray{<:Any,N}},
  axes::Tuple{Vararg{AbstractUnitRange,N}},
) where {N}
  blocks = default_blocks(block_data, axes)
  return BlockSparseArray(blocks, axes)
end

function BlockSparseArray(
  block_indices::Vector{<:Block{N}},
  block_data::Vector{<:AbstractArray{<:Any,N}},
  axes::Tuple{Vararg{AbstractUnitRange,N}},
) where {N}
  return BlockSparseArray(Dictionary(block_indices, block_data), axes)
end

function BlockSparseArray{T,N,A,Blocks}(
  blocks::AbstractArray{<:AbstractArray{T,N},N}, axes::Tuple{Vararg{AbstractUnitRange,N}}
) where {T,N,A<:AbstractArray{T,N},Blocks<:AbstractArray{A,N}}
  return BlockSparseArray{T,N,A,Blocks,typeof(axes)}(blocks, axes)
end

function BlockSparseArray{T,N,A}(
  blocks::AbstractArray{<:AbstractArray{T,N},N}, axes::Tuple{Vararg{AbstractUnitRange,N}}
) where {T,N,A<:AbstractArray{T,N}}
  return BlockSparseArray{T,N,A,typeof(blocks)}(blocks, axes)
end

function BlockSparseArray{T,N}(
  blocks::AbstractArray{<:AbstractArray{T,N},N}, axes::Tuple{Vararg{AbstractUnitRange,N}}
) where {T,N}
  return BlockSparseArray{T,N,eltype(blocks),typeof(blocks),typeof(axes)}(blocks, axes)
end

function BlockSparseArray{T,N}(
  block_data::Dictionary{Block{N,Int},<:AbstractArray{T,N}},
  axes::Tuple{Vararg{AbstractUnitRange,N}},
) where {T,N}
  blocks = default_blocks(block_data, axes)
  return BlockSparseArray{T,N}(blocks, axes)
end

function BlockSparseArray{T,N,A}(
  axes::Tuple{Vararg{AbstractUnitRange,N}}
) where {T,N,A<:AbstractArray{T,N}}
  blocks = default_blocks(A, axes)
  return BlockSparseArray{T,N,A}(blocks, axes)
end

function BlockSparseArray{T,N}(axes::Tuple{Vararg{AbstractUnitRange,N}}) where {T,N}
  return BlockSparseArray{T,N,default_arraytype(T, axes)}(axes)
end

function BlockSparseArray{T,0}(axes::Tuple{}) where {T}
  return BlockSparseArray{T,0,default_arraytype(T, axes)}(axes)
end

function BlockSparseArray{T,N}(dims::Tuple{Vararg{Vector{Int},N}}) where {T,N}
  return BlockSparseArray{T,N}(blockedrange.(dims))
end

function BlockSparseArray{T}(dims::Tuple{Vararg{Vector{Int}}}) where {T}
  return BlockSparseArray{T,length(dims)}(dims)
end

function BlockSparseArray{T}(axes::Tuple{Vararg{AbstractUnitRange}}) where {T}
  return BlockSparseArray{T,length(axes)}(axes)
end

function BlockSparseArray{T}(axes::Tuple{}) where {T}
  return BlockSparseArray{T,length(axes)}(axes)
end

function BlockSparseArray{T}(dims::Vararg{Vector{Int}}) where {T}
  return BlockSparseArray{T}(dims)
end

function BlockSparseArray{T}(axes::Vararg{AbstractUnitRange}) where {T}
  return BlockSparseArray{T}(axes)
end

function BlockSparseArray{T}() where {T}
  return BlockSparseArray{T}(())
end

function BlockSparseArray{T,N,A}(
  ::UndefInitializer, dims::Tuple
) where {T,N,A<:AbstractArray{T,N}}
  return BlockSparseArray{T,N,A}(dims)
end

# undef
function BlockSparseArray{T,N}(
  ::UndefInitializer, axes::Tuple{Vararg{AbstractUnitRange,N}}
) where {T,N}
  return BlockSparseArray{T,N}(axes)
end

function BlockSparseArray{T,N}(
  ::UndefInitializer, dims::Tuple{Vararg{Vector{Int},N}}
) where {T,N}
  return BlockSparseArray{T,N}(dims)
end

function BlockSparseArray{T}(
  ::UndefInitializer, axes::Tuple{Vararg{AbstractUnitRange}}
) where {T}
  return BlockSparseArray{T}(axes)
end

function BlockSparseArray{T}(::UndefInitializer, dims::Tuple{Vararg{Vector{Int}}}) where {T}
  return BlockSparseArray{T}(dims)
end

function BlockSparseArray{T}(::UndefInitializer, dims::Vararg{Vector{Int}}) where {T}
  return BlockSparseArray{T}(dims...)
end

# Base `AbstractArray` interface
Base.axes(a::BlockSparseArray) = a.axes

# BlockArrays `AbstractBlockArray` interface.
# This is used by `blocks(::BlockSparseArrayLike)`.
blocksparse_blocks(a::BlockSparseArray) = a.blocks

# TODO: Use `TypeParameterAccessors`.
function blockstype(
  arraytype::Type{<:BlockSparseArray{T,N,A,Blocks}}
) where {T,N,A<:AbstractArray{T,N},Blocks<:AbstractArray{A,N}}
  return Blocks
end
function blockstype(
  arraytype::Type{<:BlockSparseArray{T,N,A}}
) where {T,N,A<:AbstractArray{T,N}}
  return SparseArrayDOK{A,N}
end
function blockstype(arraytype::Type{<:BlockSparseArray{T,N}}) where {T,N}
  return SparseArrayDOK{AbstractArray{T,N},N}
end
function blockstype(arraytype::Type{<:BlockSparseArray{T}}) where {T}
  return SparseArrayDOK{AbstractArray{T}}
end
blockstype(arraytype::Type{<:BlockSparseArray}) = SparseArrayDOK{AbstractArray}

## # Base interface
## function Base.similar(
##   a::AbstractBlockSparseArray, elt::Type, axes::Tuple{Vararg{BlockedUnitRange}}
## )
##   # TODO: Preserve GPU data!
##   return BlockSparseArray{elt}(undef, axes)
## end
