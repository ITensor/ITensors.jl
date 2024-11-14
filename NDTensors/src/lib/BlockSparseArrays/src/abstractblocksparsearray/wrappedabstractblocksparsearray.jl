using Adapt: Adapt, WrappedArray
using BlockArrays:
  BlockArrays,
  AbstractBlockVector,
  AbstractBlockedUnitRange,
  BlockIndexRange,
  BlockRange,
  blockedrange,
  mortar,
  unblock
using SplitApplyCombine: groupcount
using ..TypeParameterAccessors: similartype

const WrappedAbstractBlockSparseArray{T,N} = WrappedArray{
  T,N,AbstractBlockSparseArray,AbstractBlockSparseArray{T,N}
}

# TODO: Rename `AnyBlockSparseArray`.
const AnyAbstractBlockSparseArray{T,N} = Union{
  <:AbstractBlockSparseArray{T,N},<:WrappedAbstractBlockSparseArray{T,N}
}

# a[1:2, 1:2]
function Base.to_indices(
  a::AnyAbstractBlockSparseArray, inds, I::Tuple{UnitRange{<:Integer},Vararg{Any}}
)
  return blocksparse_to_indices(a, inds, I)
end

# a[[Block(2), Block(1)], [Block(2), Block(1)]]
function Base.to_indices(
  a::AnyAbstractBlockSparseArray, inds, I::Tuple{Vector{<:Block{1}},Vararg{Any}}
)
  return blocksparse_to_indices(a, inds, I)
end

# a[BlockVector([Block(2), Block(1)], [2]), BlockVector([Block(2), Block(1)], [2])]
# a[BlockedVector([Block(2), Block(1)], [2]), BlockedVector([Block(2), Block(1)], [2])]
function Base.to_indices(
  a::AnyAbstractBlockSparseArray,
  inds,
  I::Tuple{AbstractBlockVector{<:Block{1}},Vararg{Any}},
)
  return blocksparse_to_indices(a, inds, I)
end

# a[mortar([Block(1)[1:2], Block(2)[1:3]])]
function Base.to_indices(
  a::AnyAbstractBlockSparseArray,
  inds,
  I::Tuple{BlockVector{<:BlockIndex{1},<:Vector{<:BlockIndexRange{1}}},Vararg{Any}},
)
  return blocksparse_to_indices(a, inds, I)
end

# a[[Block(1)[1:2], Block(2)[1:2]], [Block(1)[1:2], Block(2)[1:2]]]
function Base.to_indices(
  a::AnyAbstractBlockSparseArray, inds, I::Tuple{Vector{<:BlockIndexRange{1}},Vararg{Any}}
)
  return to_indices(a, inds, (mortar(I[1]), Base.tail(I)...))
end

# BlockArrays `AbstractBlockArray` interface
BlockArrays.blocks(a::AnyAbstractBlockSparseArray) = blocksparse_blocks(a)

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

blocktype(a::AnyAbstractBlockSparseArray) = eltype(blocks(a))
blocktype(arraytype::Type{<:AnyAbstractBlockSparseArray}) = eltype(blockstype(arraytype))

using ArrayLayouts: ArrayLayouts
function Base.getindex(
  a::AnyAbstractBlockSparseArray{<:Any,N}, I::CartesianIndices{N}
) where {N}
  return ArrayLayouts.layout_getindex(a, I)
end
function Base.getindex(
  a::AnyAbstractBlockSparseArray{<:Any,N}, I::Vararg{AbstractUnitRange{<:Integer},N}
) where {N}
  return ArrayLayouts.layout_getindex(a, I...)
end
# TODO: Define `AnyBlockSparseMatrix`.
function Base.getindex(
  a::AnyAbstractBlockSparseArray{<:Any,2}, I::Vararg{AbstractUnitRange{<:Integer},2}
)
  return ArrayLayouts.layout_getindex(a, I...)
end
# Fixes ambiguity error.
function Base.getindex(a::AnyAbstractBlockSparseArray{<:Any,0})
  return ArrayLayouts.layout_getindex(a)
end

# TODO: Define `blocksparse_isassigned`.
function Base.isassigned(
  a::AnyAbstractBlockSparseArray{<:Any,N}, index::Vararg{Block{1},N}
) where {N}
  return isassigned(blocks(a), Int.(index)...)
end

# Fix ambiguity error.
function Base.isassigned(a::AnyAbstractBlockSparseArray{<:Any,0})
  return isassigned(blocks(a))
end

function Base.isassigned(a::AnyAbstractBlockSparseArray{<:Any,N}, index::Block{N}) where {N}
  return isassigned(a, Tuple(index)...)
end

# TODO: Define `blocksparse_isassigned`.
function Base.isassigned(
  a::AnyAbstractBlockSparseArray{<:Any,N}, index::Vararg{BlockIndex{1},N}
) where {N}
  b = block.(index)
  return isassigned(a, b...) && isassigned(@view(a[b...]), blockindex.(index)...)
end

function Base.setindex!(
  a::AnyAbstractBlockSparseArray{<:Any,N}, value, I::BlockIndex{N}
) where {N}
  blocksparse_setindex!(a, value, I)
  return a
end
# Fixes ambiguity error with BlockArrays.jl
function Base.setindex!(a::AnyAbstractBlockSparseArray{<:Any,1}, value, I::BlockIndex{1})
  blocksparse_setindex!(a, value, I)
  return a
end

function Base.fill!(a::AbstractBlockSparseArray, value)
  if iszero(value)
    # This drops all of the blocks.
    sparse_zero!(blocks(a))
    return a
  end
  blocksparse_fill!(a, value)
  return a
end

function Base.fill!(a::AnyAbstractBlockSparseArray, value)
  # TODO: Even if `iszero(value)`, this doesn't drop
  # blocks from `a`, and additionally allocates
  # new blocks filled with zeros, unlike
  # `fill!(a::AbstractBlockSparseArray, value)`.
  # Consider changing that behavior when possible.
  blocksparse_fill!(a, value)
  return a
end

# Needed by `BlockArrays` matrix multiplication interface
function Base.similar(
  arraytype::Type{<:AnyAbstractBlockSparseArray},
  axes::Tuple{Vararg{AbstractUnitRange{<:Integer}}},
)
  return similar(arraytype, eltype(arraytype), axes)
end

# Fixes ambiguity error.
function Base.similar(
  arraytype::Type{<:AnyAbstractBlockSparseArray}, axes::Tuple{Base.OneTo,Vararg{Base.OneTo}}
)
  return similar(arraytype, eltype(arraytype), axes)
end

# Needed by `BlockArrays` matrix multiplication interface
# TODO: This fixes an ambiguity error with `OffsetArrays.jl`, but
# is only appears to be needed in older versions of Julia like v1.6.
# Delete once we drop support for older versions of Julia.
function Base.similar(
  arraytype::Type{<:AnyAbstractBlockSparseArray},
  axes::Tuple{AbstractUnitRange{<:Integer},Vararg{AbstractUnitRange{<:Integer}}},
)
  return similar(arraytype, eltype(arraytype), axes)
end

# Fixes ambiguity error with `BlockArrays`.
function Base.similar(
  arraytype::Type{<:AnyAbstractBlockSparseArray},
  axes::Tuple{AbstractBlockedUnitRange{<:Integer},Vararg{AbstractUnitRange{<:Integer}}},
)
  return similar(arraytype, eltype(arraytype), axes)
end

# Fixes ambiguity error with `BlockArrays`.
function Base.similar(
  arraytype::Type{<:AnyAbstractBlockSparseArray},
  axes::Tuple{
    AbstractUnitRange{<:Integer},
    AbstractBlockedUnitRange{<:Integer},
    Vararg{AbstractUnitRange{<:Integer}},
  },
)
  return similar(arraytype, eltype(arraytype), axes)
end

# Needed for disambiguation
function Base.similar(
  arraytype::Type{<:AnyAbstractBlockSparseArray},
  axes::Tuple{Vararg{AbstractBlockedUnitRange{<:Integer}}},
)
  return similar(arraytype, eltype(arraytype), axes)
end

function blocksparse_similar(a, elt::Type, axes::Tuple)
  return BlockSparseArray{elt,length(axes),similartype(blocktype(a), elt, axes)}(
    undef, axes
  )
end

# Needed by `BlockArrays` matrix multiplication interface
# TODO: Define a `blocksparse_similar` function.
function Base.similar(
  arraytype::Type{<:AnyAbstractBlockSparseArray},
  elt::Type,
  axes::Tuple{Vararg{AbstractUnitRange{<:Integer}}},
)
  return blocksparse_similar(arraytype, elt, axes)
end

# TODO: Define a `blocksparse_similar` function.
function Base.similar(
  a::AnyAbstractBlockSparseArray,
  elt::Type,
  axes::Tuple{Vararg{AbstractUnitRange{<:Integer}}},
)
  return blocksparse_similar(a, elt, axes)
end

# Fixes ambiguity error.
function Base.similar(a::AnyAbstractBlockSparseArray, elt::Type, axes::Tuple{})
  return blocksparse_similar(a, elt, axes)
end

# Fixes ambiguity error with `BlockArrays`.
function Base.similar(
  a::AnyAbstractBlockSparseArray,
  elt::Type,
  axes::Tuple{
    AbstractBlockedUnitRange{<:Integer},Vararg{AbstractBlockedUnitRange{<:Integer}}
  },
)
  return blocksparse_similar(a, elt, axes)
end

# Fixes ambiguity error with `OffsetArrays`.
function Base.similar(
  a::AnyAbstractBlockSparseArray,
  elt::Type,
  axes::Tuple{AbstractUnitRange{<:Integer},Vararg{AbstractUnitRange{<:Integer}}},
)
  return blocksparse_similar(a, elt, axes)
end

# Fixes ambiguity error with `BlockArrays`.
function Base.similar(
  a::AnyAbstractBlockSparseArray,
  elt::Type,
  axes::Tuple{AbstractBlockedUnitRange{<:Integer},Vararg{AbstractUnitRange{<:Integer}}},
)
  return blocksparse_similar(a, elt, axes)
end

# Fixes ambiguity errors with BlockArrays.
function Base.similar(
  a::AnyAbstractBlockSparseArray,
  elt::Type,
  axes::Tuple{
    AbstractUnitRange{<:Integer},
    AbstractBlockedUnitRange{<:Integer},
    Vararg{AbstractUnitRange{<:Integer}},
  },
)
  return blocksparse_similar(a, elt, axes)
end

# Fixes ambiguity error with `StaticArrays`.
function Base.similar(
  a::AnyAbstractBlockSparseArray, elt::Type, axes::Tuple{Base.OneTo,Vararg{Base.OneTo}}
)
  return blocksparse_similar(a, elt, axes)
end

# TODO: Implement this in a more generic way using a smarter `copyto!`,
# which is ultimately what `Array{T,N}(::AbstractArray{<:Any,N})` calls.
# These are defined for now to avoid scalar indexing issues when there
# are blocks on GPU.
function Base.Array{T,N}(a::AnyAbstractBlockSparseArray{<:Any,N}) where {T,N}
  # First make it dense, then move to CPU.
  # Directly copying to CPU causes some issues with
  # scalar indexing on GPU which we have to investigate.
  a_dest = similartype(blocktype(a), T)(undef, size(a))
  a_dest .= a
  return Array{T,N}(a_dest)
end
function Base.Array{T}(a::AnyAbstractBlockSparseArray) where {T}
  return Array{T,ndims(a)}(a)
end
function Base.Array(a::AnyAbstractBlockSparseArray)
  return Array{eltype(a)}(a)
end
