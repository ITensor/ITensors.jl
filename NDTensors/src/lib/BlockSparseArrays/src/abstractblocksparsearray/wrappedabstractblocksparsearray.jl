using Adapt: Adapt, WrappedArray
using BlockArrays:
  BlockArrays, BlockedUnitRange, BlockIndexRange, BlockRange, blockedrange, mortar, unblock
using SplitApplyCombine: groupcount

const WrappedAbstractBlockSparseArray{T,N} = WrappedArray{
  T,N,AbstractBlockSparseArray,AbstractBlockSparseArray{T,N}
}

# TODO: Rename `AnyBlockSparseArray`.
const BlockSparseArrayLike{T,N} = Union{
  <:AbstractBlockSparseArray{T,N},<:WrappedAbstractBlockSparseArray{T,N}
}

# Used when making views.
# TODO: Move to blocksparsearrayinterface.
function blocksparse_to_indices(a, inds, I)
  return (unblock(a, inds, I), to_indices(a, BlockArrays._maybetail(inds), Base.tail(I))...)
end

# TODO: Move to blocksparsearrayinterface.
function blocksparse_to_indices(a, I)
  return to_indices(a, axes(a), I)
end

# Used when making views.
function Base.to_indices(
  a::BlockSparseArrayLike, inds, I::Tuple{AbstractVector{<:Block{1}},Vararg{Any}}
)
  return blocksparse_to_indices(a, inds, I)
end

function Base.to_indices(
  a::BlockSparseArrayLike, inds, I::Tuple{AbstractUnitRange{<:Integer},Vararg{Any}}
)
  return blocksparse_to_indices(a, inds, I)
end

# Fixes ambiguity error with BlockArrays.
function Base.to_indices(a::BlockSparseArrayLike, inds, I::Tuple{BlockRange{1},Vararg{Any}})
  return blocksparse_to_indices(a, inds, I)
end

function Base.to_indices(
  a::BlockSparseArrayLike, I::Tuple{AbstractVector{<:Block{1}},Vararg{Any}}
)
  return blocksparse_to_indices(a, I)
end

# Handle case of indexing with `[Block(1)[1:2], Block(2)[1:2]]`
# by converting it to a `BlockVector` with
# `mortar([Block(1)[1:2], Block(2)[1:2]])`.
function Base.to_indices(
  a::BlockSparseArrayLike, inds, I::Tuple{AbstractVector{<:BlockIndexRange{1}},Vararg{Any}}
)
  return to_indices(a, inds, (mortar(I[1]), Base.tail(I)...))
end

# Fixes ambiguity error with BlockArrays.
function Base.to_indices(a::BlockSparseArrayLike, I::Tuple{BlockRange{1},Vararg{Any}})
  return blocksparse_to_indices(a, I)
end

function Base.to_indices(
  a::BlockSparseArrayLike, I::Tuple{AbstractUnitRange{<:Integer},Vararg{Any}}
)
  return blocksparse_to_indices(a, I)
end

# Used inside `Base.to_indices` when making views.
# TODO: Move to blocksparsearrayinterface.
# TODO: Make a special definition for `BlockedVector{<:Block{1}}` in order
# to merge blocks.
function blocksparse_unblock(a, inds, I::Tuple{AbstractVector{<:Block{1}},Vararg{Any}})
  return BlockIndices(I[1], blockedunitrange_getindices(inds[1], I[1]))
end

# TODO: Move to blocksparsearrayinterface.
function blocksparse_unblock(a, inds, I::Tuple{AbstractUnitRange{<:Integer},Vararg{Any}})
  bs = blockrange(inds[1], I[1])
  return BlockSlice(bs, blockedunitrange_getindices(inds[1], I[1]))
end

function BlockArrays.unblock(a, inds, I::Tuple{AbstractVector{<:Block{1}},Vararg{Any}})
  return blocksparse_unblock(a, inds, I)
end

function BlockArrays.unblock(
  a::BlockSparseArrayLike, inds, I::Tuple{AbstractUnitRange{<:Integer},Vararg{Any}}
)
  return blocksparse_unblock(a, inds, I)
end

# BlockArrays `AbstractBlockArray` interface
BlockArrays.blocks(a::BlockSparseArrayLike) = blocksparse_blocks(a)

# Fix ambiguity error with `BlockArrays`
using BlockArrays: BlockSlice
function BlockArrays.blocks(
  a::SubArray{<:Any,<:Any,<:AbstractBlockSparseArray,<:Tuple{Vararg{BlockSlice}}}
)
  return blocksparse_blocks(a)
end

using ..TypeParameterAccessors: parenttype
function blockstype(arraytype::Type{<:WrappedAbstractBlockSparseArray})
  return blockstype(parenttype(arraytype))
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

function Base.getindex(a::BlockSparseArrayLike{<:Any,N}, block::Block{N}) where {N}
  return blocksparse_getindex(a, block)
end
function Base.getindex(
  a::BlockSparseArrayLike{<:Any,N}, block::Vararg{Block{1},N}
) where {N}
  return blocksparse_getindex(a, block...)
end

# TODO: Define `blocksparse_isassigned`.
function Base.isassigned(
  a::BlockSparseArrayLike{<:Any,N}, index::Vararg{Block{1},N}
) where {N}
  return isassigned(blocks(a), Int.(index)...)
end

function Base.isassigned(a::BlockSparseArrayLike{<:Any,N}, index::Block{N}) where {N}
  return isassigned(a, Tuple(index)...)
end

# TODO: Define `blocksparse_isassigned`.
function Base.isassigned(
  a::BlockSparseArrayLike{<:Any,N}, index::Vararg{BlockIndex{1},N}
) where {N}
  b = block.(index)
  return isassigned(a, b...) && isassigned(@view(a[b...]), blockindex.(index)...)
end

function Base.setindex!(a::BlockSparseArrayLike{<:Any,N}, value, I::BlockIndex{N}) where {N}
  blocksparse_setindex!(a, value, I)
  return a
end

function Base.setindex!(
  a::BlockSparseArrayLike{<:Any,N}, value, I::Vararg{Block{1},N}
) where {N}
  a[Block(Int.(I))] = value
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

# Needed by `BlockArrays` matrix multiplication interface
# TODO: This fixes an ambiguity error with `OffsetArrays.jl`, but
# is only appears to be needed in older versions of Julia like v1.6.
# Delete once we drop support for older versions of Julia.
function Base.similar(
  arraytype::Type{<:BlockSparseArrayLike},
  axes::Tuple{AbstractUnitRange,Vararg{AbstractUnitRange}},
)
  return similar(arraytype, eltype(arraytype), axes)
end

# Needed by `BlockArrays` matrix multiplication interface
# Fixes ambiguity error with `BlockArrays.jl`.
function Base.similar(
  arraytype::Type{<:BlockSparseArrayLike},
  axes::Tuple{BlockedUnitRange,Vararg{AbstractUnitRange{Int}}},
)
  return similar(arraytype, eltype(arraytype), axes)
end

# Needed by `BlockArrays` matrix multiplication interface
# Fixes ambiguity error with `BlockArrays.jl`.
function Base.similar(
  arraytype::Type{<:BlockSparseArrayLike},
  axes::Tuple{AbstractUnitRange{Int},BlockedUnitRange,Vararg{AbstractUnitRange{Int}}},
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
