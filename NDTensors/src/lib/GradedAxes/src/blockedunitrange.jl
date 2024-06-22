using BlockArrays:
  BlockArrays,
  AbstractBlockedUnitRange,
  Block,
  BlockIndexRange,
  BlockRange,
  BlockSlice,
  BlockedUnitRange,
  block,
  blockindex,
  findblock,
  findblockindex

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
function blockedunitrange_findblock(a::AbstractBlockedUnitRange, index::Integer)
  @boundscheck index in 1:length(a) || throw(BoundsError(a, index))
  return @inbounds findblock(a, index + first(a) - 1)
end

# TODO: Move this to a `BlockArraysExtensions` library.
# TODO: Rename this. `BlockArrays.findblockindex(a, k)` finds the
# block index of the value `k`, while this finds the block index of the index `k`.
# This could make use of the `BlockIndices` object, i.e. `BlockIndices(a)[index]`.
function blockedunitrange_findblockindex(a::AbstractBlockedUnitRange, index::Integer)
  @boundscheck index in 1:length(a) || throw(BoundsError())
  return @inbounds findblockindex(a, index + first(a) - 1)
end

function blockedunitrange_getindices(a::AbstractUnitRange, indices)
  return a[indices]
end

# TODO: Move this to a `BlockArraysExtensions` library.
# Like `a[indices]` but preserves block structure.
# TODO: Consider calling this something else, for example
# `blocked_getindex`. See the discussion here:
# https://github.com/JuliaArrays/BlockArrays.jl/issues/347
function blockedunitrange_getindices(
  a::AbstractBlockedUnitRange, indices::AbstractUnitRange{<:Integer}
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
function blockedunitrange_getindices(a::AbstractBlockedUnitRange, indices::BlockIndexRange)
  return a[block(indices)][only(indices.indices)]
end

# TODO: Move this to a `BlockArraysExtensions` library.
function blockedunitrange_getindices(a::AbstractBlockedUnitRange, indices::BlockSlice)
  # TODO: Is this a good definition? It ignores `indices.indices`.
  return a[indices.block]
end

# TODO: Move this to a `BlockArraysExtensions` library.
function blockedunitrange_getindices(
  a::AbstractBlockedUnitRange, indices::Vector{<:Integer}
)
  return map(index -> a[index], indices)
end

# TODO: Move this to a `BlockArraysExtensions` library.
# TODO: Make a special definition for `BlockedVector{<:Block{1}}` in order
# to merge blocks.
function blockedunitrange_getindices(
  a::AbstractBlockedUnitRange, indices::AbstractVector{<:Union{Block{1},BlockIndexRange{1}}}
)
  # Without converting `indices` to `Vector`,
  # mapping `indices` outputs a `BlockVector`
  # which is harder to reason about.
  blocks = map(index -> a[index], Vector(indices))
  # We pass `length.(blocks)` to `mortar` in order
  # to pass block labels to the axes of the output,
  # if they exist. This makes it so that
  # `only(axes(a[indices])) isa `GradedUnitRange`
  # if `a isa `GradedUnitRange`, for example.
  return mortar(blocks, length.(blocks))
end

# TODO: Move this to a `BlockArraysExtensions` library.
function blockedunitrange_getindices(a::AbstractBlockedUnitRange, indices::Block{1})
  return a[indices]
end

# TODO: Move this to a `BlockArraysExtensions` library.
function blockedunitrange_getindices(a::AbstractBlockedUnitRange, indices)
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
