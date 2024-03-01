using BlockArrays: BlockedUnitRange, blockedrange
using SplitApplyCombine: groupcount

## # TODO: Write a specialized version for `indices::AbstractUnitRange`.
## # TODO: Write a generic version for blocked unit ranges (like `GradedAxes`).
## function sub_unitrange(a::BlockedUnitRange, indices)
##   indices = sort(indices)
##   return blockedrange(collect(groupcount(i -> findblock(a, i), indices)))
## end

using Adapt: Adapt, WrappedArray

const WrappedAbstractBlockSparseArray{T,N,A} = WrappedArray{
  T,N,<:AbstractBlockSparseArray,<:AbstractBlockSparseArray{T,N}
}

const BlockSparseArrayLike{T,N} = Union{
  <:AbstractBlockSparseArray{T,N},<:WrappedAbstractBlockSparseArray{T,N}
}

# AbstractArray interface
# TODO: Use `BlockSparseArrayLike`.
# TODO: Need to handle block indexing.
function Base.axes(a::SubArray{<:Any,<:Any,<:AbstractBlockSparseArray})
  return ntuple(i -> sub_unitrange(axes(parent(a), i), a.indices[i]), ndims(a))
end

# BlockArrays `AbstractBlockArray` interface
BlockArrays.blocks(a::BlockSparseArrayLike) = blocksparse_blocks(a)

# TODO: Use `TypeParameterAccessors`.
function blockstype(arraytype::Type{<:WrappedAbstractBlockSparseArray})
  return blockstype(Adapt.parent_type(arraytype))
end

blocktype(a::BlockSparseArrayLike) = eltype(blocks(a))
blocktype(arraytype::Type{<:BlockSparseArrayLike}) = eltype(blockstype(arraytype))

using ArrayLayouts: ArrayLayouts
## function Base.getindex(a::BlockSparseArrayLike{<:Any,N}, I::Vararg{Int,N}) where {N}
##   return ArrayLayouts.layout_getindex(a, I...)
## end
function Base.getindex(a::BlockSparseArrayLike{<:Any,N}, I::CartesianIndices{N}) where {N}
  return ArrayLayouts.layout_getindex(a, I)
end
function Base.getindex(
  a::BlockSparseArrayLike{<:Any,N}, I::Vararg{AbstractUnitRange,N}
) where {N}
  return ArrayLayouts.layout_getindex(a, I...)
end
# TODO: Define `AnyBlockSparseMatrix`.
function Base.getindex(a::BlockSparseArrayLike{<:Any,2}, I::Vararg{AbstractUnitRange,2})
  return ArrayLayouts.layout_getindex(a, I...)
end

function Base.setindex!(a::BlockSparseArrayLike{<:Any,N}, value, I::BlockIndex{N}) where {N}
  blocksparse_setindex!(a, value, I)
  return a
end

function Base.setindex!(a::BlockSparseArrayLike{<:Any,N}, value, I::Block{N}) where {N}
  blocksparse_setindex!(a, value, I)
  return a
end

# Fix ambiguity error
function Base.setindex!(a::BlockSparseArrayLike{<:Any,1}, value, I::Block{1})
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
  arraytype::Type{<:BlockSparseArrayLike}, axes::Tuple{Vararg{AbstractUnitRange}}
)
  return similar(arraytype, eltype(arraytype), axes)
end

# Needed for disambiguation
function Base.similar(
  arraytype::Type{<:BlockSparseArrayLike}, axes::Tuple{Vararg{BlockedUnitRange}}
)
  return similar(arraytype, eltype(arraytype), axes)
end

# Needed by `BlockArrays` matrix multiplication interface
# TODO: Define a `blocksparse_similar` function.
function Base.similar(
  arraytype::Type{<:BlockSparseArrayLike}, elt::Type, axes::Tuple{Vararg{AbstractUnitRange}}
)
  # TODO: Make generic for GPU, maybe using `blocktype`.
  # TODO: For non-block axes this should output `Array`.
  return BlockSparseArray{elt}(undef, axes)
end

# TODO: Define a `blocksparse_similar` function.
function Base.similar(
  a::BlockSparseArrayLike, elt::Type, axes::Tuple{Vararg{AbstractUnitRange}}
)
  # TODO: Make generic for GPU, maybe using `blocktype`.
  # TODO: For non-block axes this should output `Array`.
  return BlockSparseArray{elt}(undef, axes)
end

# TODO: Define a `blocksparse_similar` function.
# Fixes ambiguity error with `BlockArrays`.
function Base.similar(
  a::BlockSparseArrayLike, elt::Type, axes::Tuple{BlockedUnitRange,Vararg{BlockedUnitRange}}
)
  # TODO: Make generic for GPU, maybe using `blocktype`.
  # TODO: For non-block axes this should output `Array`.
  return BlockSparseArray{elt}(undef, axes)
end

# TODO: Define a `blocksparse_similar` function.
# Fixes ambiguity error with `OffsetArrays`.
function Base.similar(
  a::BlockSparseArrayLike,
  elt::Type,
  axes::Tuple{AbstractUnitRange,Vararg{AbstractUnitRange}},
)
  # TODO: Make generic for GPU, maybe using `blocktype`.
  # TODO: For non-block axes this should output `Array`.
  return BlockSparseArray{elt}(undef, axes)
end

# TODO: Define a `blocksparse_similar` function.
# Fixes ambiguity error with `StaticArrays`.
function Base.similar(
  a::BlockSparseArrayLike, elt::Type, axes::Tuple{Base.OneTo,Vararg{Base.OneTo}}
)
  # TODO: Make generic for GPU, maybe using `blocktype`.
  # TODO: For non-block axes this should output `Array`.
  return BlockSparseArray{elt}(undef, axes)
end
