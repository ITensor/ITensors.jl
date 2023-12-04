using BlockArrays: BlockArrays, AbstractBlockArray, Block, BlockIndex, BlockedUnitRange

# TODO: Delete this. This function was replaced
# by `nstored` but is still used in `NDTensors`.
function nonzero_keys end

abstract type AbstractBlockSparseArray{T,N} <: AbstractBlockArray{T,N} end

# Base `AbstractArray` interface
Base.axes(::AbstractBlockSparseArray) = error("Not implemented")

# BlockArrays `AbstractBlockArray` interface
BlockArrays.blocks(::AbstractBlockSparseArray) = error("Not implemented")

blocktype(a::AbstractBlockSparseArray) = eltype(blocks(a))

blockstype(::Type{<:AbstractBlockSparseArray}) = error("Not implemented")
blocktype(arraytype::Type{<:AbstractBlockSparseArray}) = eltype(blockstype(arraytype))

# Base `AbstractArray` interface
function Base.getindex(a::AbstractBlockSparseArray{<:Any,N}, I::Vararg{Int,N}) where {N}
  return blocksparse_getindex(a, I...)
end

function Base.setindex!(
  a::AbstractBlockSparseArray{<:Any,N}, value, I::Vararg{Int,N}
) where {N}
  blocksparse_setindex!(a, value, I...)
  return a
end

function Base.setindex!(
  a::AbstractBlockSparseArray{<:Any,N}, value, I::BlockIndex{N}
) where {N}
  blocksparse_setindex!(a, value, I)
  return a
end

function Base.setindex!(a::AbstractBlockSparseArray{<:Any,N}, value, I::Block{N}) where {N}
  blocksparse_setindex!(a, value, I)
  return a
end

# `BlockArrays` interface
function BlockArrays.viewblock(
  a::AbstractBlockSparseArray{<:Any,N}, I::Block{N,Int}
) where {N}
  return blocksparse_viewblock(a, I)
end

# Needed by `BlockArrays` matrix multiplication interface
function Base.similar(
  arraytype::Type{<:AbstractBlockSparseArray{T}}, axes::Tuple{Vararg{BlockedUnitRange}}
) where {T}
  # TODO: Make generic for GPU!
  return BlockSparseArray{T}(undef, axes)
end
