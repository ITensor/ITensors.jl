using BlockArrays:
  BlockArrays,
  Block,
  BlockedUnitRange,
  BlockIndex,
  BlockRange,
  BlockVector,
  blockedrange,
  BlockIndexRange,
  blockfirsts,
  blocklasts,
  blocklength,
  blocklengths,
  findblock,
  findblockindex,
  mortar
using ..LabelledNumbers: LabelledNumbers, LabelledInteger, label, labelled, unlabel

# Custom `BlockedUnitRange` constructor that takes a unit range
# and a set of block lengths, similar to `BlockArray(::AbstractArray, blocklengths...)`.
function blockedunitrange(a::AbstractUnitRange, blocklengths)
  blocklengths_shifted = copy(blocklengths)
  blocklengths_shifted[1] += (first(a) - 1)
  blocklasts = cumsum(blocklengths_shifted)
  return BlockArrays._BlockedUnitRange(first(a), blocklasts)
end

# TODO: Move this to a `BlockArraysExtensions` library.
# TODO: Rename this. `BlockArrays.findblock(a, k)` finds the
# block of the value `k`, while this finds the block of the index `k`.
# This could make use of the `BlockIndices` object, i.e. `block(BlockIndices(a)[index])`.
function blockedunitrange_findblock(a::BlockedUnitRange, index::Integer)
  @boundscheck index in 1:length(a) || throw(BoundsError(a, index))
  return @inbounds findblock(a, index + first(a) - 1)
end

# TODO: Move this to a `BlockArraysExtensions` library.
# TODO: Rename this. `BlockArrays.findblockindex(a, k)` finds the
# block index of the value `k`, while this finds the block index of the index `k`.
# This could make use of the `BlockIndices` object, i.e. `BlockIndices(a)[index]`.
function blockedunitrange_findblockindex(a::BlockedUnitRange, index::Integer)
  @boundscheck index in 1:length(a) || throw(BoundsError())
  return @inbounds findblockindex(a, index + first(a) - 1)
end

const GradedUnitRange{BlockLasts<:Vector{<:LabelledInteger}} = BlockedUnitRange{BlockLasts}

# TODO: Use `TypeParameterAccessors`.
Base.eltype(::Type{<:GradedUnitRange{<:Vector{T}}}) where {T} = T

function gradedrange(lblocklengths::AbstractVector{<:LabelledInteger})
  brange = blockedrange(unlabel.(lblocklengths))
  lblocklasts = labelled.(blocklasts(brange), label.(lblocklengths))
  # TODO: `first` is forced to be `Int` in `BlockArrays.BlockedUnitRange`,
  # so this doesn't do anything right now. Make a PR to generalize it.
  firstlength = first(lblocklengths)
  lfirst = oneunit(firstlength)
  return BlockArrays._BlockedUnitRange(lfirst, lblocklasts)
end

# To help with generic code.
function BlockArrays.blockedrange(lblocklengths::AbstractVector{<:LabelledInteger})
  return gradedrange(lblocklengths)
end

Base.last(a::GradedUnitRange) = isempty(a.lasts) ? first(a) - 1 : last(a.lasts)

# TODO: This needs to be defined to circumvent an issue
# in the `BlockArrays.BlocksView` constructor. This
# is likely caused by issues around `BlockedUnitRange` constraining
# the element type to be `Int`, which is being fixed in:
# https://github.com/JuliaArrays/BlockArrays.jl/pull/337
# Remove this definition once that is fixed.
function BlockArrays.blocks(a::GradedUnitRange)
  # TODO: Fix `BlockRange`, try using `BlockRange` instead.
  return [a[Block(i)] for i in 1:blocklength(a)]
end

function gradedrange(lblocklengths::AbstractVector{<:Pair{<:Any,<:Integer}})
  return gradedrange(labelled.(last.(lblocklengths), first.(lblocklengths)))
end

function labelled_blocks(a::BlockedUnitRange, labels)
  return BlockArrays._BlockedUnitRange(a.first, labelled.(a.lasts, labels))
end

function BlockArrays.findblock(a::GradedUnitRange, index::Integer)
  return blockedunitrange_findblock(unlabel_blocks(a), index)
end

function blockedunitrange_findblock(a::GradedUnitRange, index::Integer)
  return blockedunitrange_findblock(unlabel_blocks(a), index)
end

function blockedunitrange_findblockindex(a::GradedUnitRange, index::Integer)
  return blockedunitrange_findblockindex(unlabel_blocks(a), index)
end

function BlockArrays.findblockindex(a::GradedUnitRange, index::Integer)
  return blockedunitrange_findblockindex(unlabel_blocks(a), index)
end

## Block label interface

# Internal function
function get_label(a::BlockedUnitRange, index::Block{1})
  return label(blocklasts(a)[Int(index)])
end

# Internal function
function get_label(a::BlockedUnitRange, index::Integer)
  return get_label(a, blockedunitrange_findblock(a, index))
end

function blocklabels(a::BlockVector)
  return map(BlockRange(a)) do block
    return label(@view(a[block]))
  end
end

function blocklabels(a::BlockedUnitRange)
  # Using `a.lasts` here since that is what is stored
  # inside of `BlockedUnitRange`, maybe change that.
  # For example, it could be something like:
  #
  # map(BlockRange(a)) do block
  #   return label(@view(a[block]))
  # end
  #
  return label.(a.lasts)
end

# TODO: This relies on internals of `BlockArrays`, maybe redesign
# to try to avoid that.
# TODO: Define `set_grades`, `set_sector_labels`, `set_labels`.
function unlabel_blocks(a::BlockedUnitRange)
  return BlockArrays._BlockedUnitRange(a.first, unlabel.(a.lasts))
end

## BlockedUnitRage interface

function Base.axes(ga::GradedUnitRange)
  return map(axes(unlabel_blocks(ga))) do a
    return labelled_blocks(a, blocklabels(ga))
  end
end

function BlockArrays.blockfirsts(a::GradedUnitRange)
  return labelled.(blockfirsts(unlabel_blocks(a)), blocklabels(a))
end

function BlockArrays.blocklasts(a::GradedUnitRange)
  return labelled.(blocklasts(unlabel_blocks(a)), blocklabels(a))
end

function BlockArrays.blocklengths(a::GradedUnitRange)
  return labelled.(blocklengths(unlabel_blocks(a)), blocklabels(a))
end

function Base.first(a::GradedUnitRange)
  return labelled(first(unlabel_blocks(a)), label(a[Block(1)]))
end

Base.iterate(a::GradedUnitRange) = isempty(a) ? nothing : (first(a), first(a))
function Base.iterate(a::GradedUnitRange, i)
  i == last(a) && return nothing
  next = a[i + step(a)]
  return (next, next)
end

function firstblockindices(a::GradedUnitRange)
  return labelled.(firstblockindices(unlabel_blocks(a)), blocklabels(a))
end

function blockedunitrange_getindex(a::GradedUnitRange, index)
  # This uses `blocklasts` since that is what is stored
  # in `BlockedUnitRange`, maybe abstract that away.
  return labelled(unlabel_blocks(a)[index], get_label(a, index))
end

# TODO: Move this to a `BlockArraysExtensions` library.
# Like `a[indices]` but preserves block structure.
using BlockArrays: block, blockindex
function blockedunitrange_getindices(
  a::BlockedUnitRange, indices::AbstractUnitRange{<:Integer}
)
  first_blockindex = blockedunitrange_findblockindex(a, first(indices))
  last_blockindex = blockedunitrange_findblockindex(a, last(indices))
  first_block = block(first_blockindex)
  last_block = block(last_blockindex)
  blocklengths = if first_block == last_block
    [length(indices)]
  else
    map(first_block:last_block) do block
      if block == first_block
        return length(a[first_block]) - blockindex(first_blockindex) + 1
      end
      if block == last_block
        return blockindex(last_blockindex)
      end
      return length(a[block])
    end
  end
  return blockedunitrange(indices .+ (first(a) - 1), blocklengths)
end

# TODO: Move this to a `BlockArraysExtensions` library.
function blockedunitrange_getindices(a::BlockedUnitRange, indices::BlockIndexRange)
  return a[block(indices)][only(indices.indices)]
end

# TODO: Move this to a `BlockArraysExtensions` library.
function blockedunitrange_getindices(a::BlockedUnitRange, indices::Vector{<:Integer})
  return map(index -> a[index], indices)
end

# TODO: Move this to a `BlockArraysExtensions` library.
function blockedunitrange_getindices(
  a::BlockedUnitRange, indices::Vector{<:Union{Block{1},BlockIndexRange{1}}}
)
  return mortar(map(index -> a[index], indices))
end

# TODO: Move this to a `BlockArraysExtensions` library.
function blockedunitrange_getindices(a::BlockedUnitRange, indices)
  return error("Not implemented.")
end

# The blocks of the corresponding slice.
_blocks(a::AbstractUnitRange, indices) = error("Not implemented")
function _blocks(a::AbstractUnitRange, indices::AbstractUnitRange)
  return findblock(a, first(indices)):findblock(a, last(indices))
end
function _blocks(a::AbstractUnitRange, indices::BlockRange)
  return indices
end

# The block labels of the corresponding slice.
function blocklabels(a::AbstractUnitRange, indices)
  return map(_blocks(a, indices)) do block
    return label(a[block])
  end
end

function blockedunitrange_getindices(
  ga::GradedUnitRange, indices::AbstractUnitRange{<:Integer}
)
  a_indices = blockedunitrange_getindices(unlabel_blocks(ga), indices)
  return labelled_blocks(a_indices, blocklabels(ga, indices))
end

function blockedunitrange_getindices(ga::GradedUnitRange, indices::BlockRange)
  return labelled_blocks(unlabel_blocks(ga)[indices], blocklabels(ga, indices))
end

function blockedunitrange_getindices(a::GradedUnitRange, indices::BlockIndex{1})
  return a[block(indices)][blockindex(indices)]
end

function Base.getindex(a::GradedUnitRange, index::Integer)
  return blockedunitrange_getindex(a, index)
end

function Base.getindex(a::GradedUnitRange, index::Block{1})
  return blockedunitrange_getindex(a, index)
end

function Base.getindex(a::GradedUnitRange, indices::BlockIndexRange)
  return blockedunitrange_getindices(a, indices)
end

function Base.getindex(
  a::GradedUnitRange, indices::BlockRange{1,<:Tuple{AbstractUnitRange{Int}}}
)
  return blockedunitrange_getindices(a, indices)
end

# Fixes ambiguity error with `BlockArrays`.
function Base.getindex(a::GradedUnitRange, indices::BlockRange{1,Tuple{Base.OneTo{Int}}})
  return blockedunitrange_getindices(a, indices)
end

function Base.getindex(a::GradedUnitRange, indices::BlockIndex{1})
  return blockedunitrange_getindices(a, indices)
end

function Base.getindex(a::GradedUnitRange, indices)
  return blockedunitrange_getindices(a, indices)
end

function Base.getindex(a::GradedUnitRange, indices::AbstractUnitRange{<:Integer})
  return blockedunitrange_getindices(a, indices)
end
