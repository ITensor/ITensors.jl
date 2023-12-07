using Adapt: WrappedArray

const WrappedAbstractBlockSparseArray{T,N,A} = WrappedArray{
  T,N,<:AbstractBlockSparseArray{T,N}
}

const BlockSparseArrayLike{T,N} = Union{
  <:AbstractBlockSparseArray{T,N},<:WrappedAbstractBlockSparseArray{T,N}
}

# BlockArrays `AbstractBlockArray` interface
BlockArrays.blocks(a::BlockSparseArrayLike) = blocksparse_blocks(a)

blocktype(a::BlockSparseArrayLike) = eltype(blocks(a))

# TODO: Use `parenttype` from `Unwrap`.
blockstype(arraytype::Type{<:WrappedAbstractBlockSparseArray}) = parenttype(arraytype)

blocktype(arraytype::Type{<:BlockSparseArrayLike}) = eltype(blockstype(arraytype))

function Base.setindex!(a::BlockSparseArrayLike{<:Any,N}, value, I::BlockIndex{N}) where {N}
  blocksparse_setindex!(a, value, I)
  return a
end

function Base.setindex!(a::BlockSparseArrayLike{<:Any,N}, value, I::Block{N}) where {N}
  blocksparse_setindex!(a, value, I)
  return a
end

# `BlockArrays` interface
# TODO: Is this needed if `blocks` is defined?
function BlockArrays.viewblock(a::BlockSparseArrayLike{<:Any,N}, I::Block{N,Int}) where {N}
  return blocksparse_viewblock(a, I)
end

# Needed by `BlockArrays` matrix multiplication interface
function Base.similar(
  arraytype::Type{<:BlockSparseArrayLike}, axes::Tuple{Vararg{BlockedUnitRange}}
)
  return similar(arraytype, eltype(arraytype), axes)
end

# Needed by `BlockArrays` matrix multiplication interface
function Base.similar(
  arraytype::Type{<:BlockSparseArrayLike}, elt::Type, axes::Tuple{Vararg{BlockedUnitRange}}
)
  # TODO: Make generic for GPU! Use `blocktype`.
  return BlockSparseArray{elt}(undef, axes)
end

function Base.similar(
  a::BlockSparseArrayLike, elt::Type, axes::Tuple{Vararg{BlockedUnitRange}}
)
  # TODO: Make generic for GPU! Use `blocktype`.
  return BlockSparseArray{eltype(a)}(undef, axes)
end
