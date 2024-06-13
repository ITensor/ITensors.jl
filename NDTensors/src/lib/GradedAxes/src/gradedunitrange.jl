using BlockArrays:
  BlockArrays,
  AbstractBlockVector,
  AbstractBlockedUnitRange,
  Block,
  BlockIndex,
  BlockRange,
  BlockSlice,
  BlockVector,
  BlockedOneTo,
  BlockedUnitRange,
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

const AbstractGradedUnitRange{T<:LabelledInteger} = AbstractBlockedUnitRange{T}

const GradedUnitRange{T<:LabelledInteger,BlockLasts<:Vector{T}} = BlockedUnitRange{
  T,BlockLasts
}

const GradedOneTo{T<:LabelledInteger,BlockLasts<:Vector{T}} = BlockedOneTo{T,BlockLasts}

# This is only needed in certain Julia versions below 1.10
# (for example Julia 1.6).
# TODO: Delete this once we drop Julia 1.6 support.
function Base.OrdinalRange{T,T}(a::GradedOneTo{<:LabelledInteger{T}}) where {T}
  return unlabel_blocks(a)
end

# TODO: See if this is needed.
function Base.AbstractUnitRange{T}(a::GradedOneTo{<:LabelledInteger{T}}) where {T}
  return unlabel_blocks(a)
end

# TODO: Use `TypeParameterAccessors`.
Base.eltype(::Type{<:GradedUnitRange{T}}) where {T} = T

function gradedrange(lblocklengths::AbstractVector{<:LabelledInteger})
  brange = blockedrange(unlabel.(lblocklengths))
  lblocklasts = labelled.(blocklasts(brange), label.(lblocklengths))
  return BlockedOneTo(lblocklasts)
end

# To help with generic code.
function BlockArrays.blockedrange(lblocklengths::AbstractVector{<:LabelledInteger})
  return gradedrange(lblocklengths)
end

Base.last(a::AbstractGradedUnitRange) = isempty(a.lasts) ? first(a) - 1 : last(a.lasts)

# TODO: This needs to be defined to circumvent an issue
# in the `BlockArrays.BlocksView` constructor. This
# is likely caused by issues around `BlockedUnitRange` constraining
# the element type to be `Int`, which is being fixed in:
# https://github.com/JuliaArrays/BlockArrays.jl/pull/337
# Remove this definition once that is fixed.
function BlockArrays.blocks(a::AbstractGradedUnitRange)
  # TODO: Fix `BlockRange`, try using `BlockRange` instead.
  return [a[Block(i)] for i in 1:blocklength(a)]
end

function gradedrange(lblocklengths::AbstractVector{<:Pair{<:Any,<:Integer}})
  return gradedrange(labelled.(last.(lblocklengths), first.(lblocklengths)))
end

function labelled_blocks(a::BlockedOneTo, labels)
  # TODO: Use `blocklasts(a)`? That might
  # cause a recursive loop.
  return BlockedOneTo(labelled.(a.lasts, labels))
end
function labelled_blocks(a::BlockedUnitRange, labels)
  # TODO: Use `first(a)` and `blocklasts(a)`? Those might
  # cause a recursive loop.
  return BlockArrays._BlockedUnitRange(
    labelled(a.first, labels[1]), labelled.(a.lasts, labels)
  )
end

function BlockArrays.findblock(a::AbstractGradedUnitRange, index::Integer)
  return blockedunitrange_findblock(unlabel_blocks(a), index)
end

function blockedunitrange_findblock(a::AbstractGradedUnitRange, index::Integer)
  return blockedunitrange_findblock(unlabel_blocks(a), index)
end

function blockedunitrange_findblockindex(a::AbstractGradedUnitRange, index::Integer)
  return blockedunitrange_findblockindex(unlabel_blocks(a), index)
end

function BlockArrays.findblockindex(a::AbstractGradedUnitRange, index::Integer)
  return blockedunitrange_findblockindex(unlabel_blocks(a), index)
end

## Block label interface

# Internal function
function get_label(a::AbstractUnitRange, index::Block{1})
  return label(blocklasts(a)[Int(index)])
end

# Internal function
function get_label(a::AbstractUnitRange, index::Integer)
  return get_label(a, blockedunitrange_findblock(a, index))
end

function blocklabels(a::AbstractBlockVector)
  return map(BlockRange(a)) do block
    return label(@view(a[block]))
  end
end

function blocklabels(a::AbstractBlockedUnitRange)
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
function unlabel_blocks(a::BlockedOneTo)
  # TODO: Use `blocklasts(a)`.
  return BlockedOneTo(unlabel.(a.lasts))
end
function unlabel_blocks(a::BlockedUnitRange)
  return BlockArrays._BlockedUnitRange(a.first, unlabel.(a.lasts))
end

## BlockedUnitRage interface

function Base.axes(ga::AbstractGradedUnitRange)
  return map(axes(unlabel_blocks(ga))) do a
    return labelled_blocks(a, blocklabels(ga))
  end
end

function gradedunitrange_blockfirsts(a::AbstractGradedUnitRange)
  return labelled.(blockfirsts(unlabel_blocks(a)), blocklabels(a))
end
function BlockArrays.blockfirsts(a::AbstractGradedUnitRange)
  return gradedunitrange_blockfirsts(a)
end
function BlockArrays.blockfirsts(a::GradedOneTo)
  return gradedunitrange_blockfirsts(a)
end

function BlockArrays.blocklasts(a::AbstractGradedUnitRange)
  return labelled.(blocklasts(unlabel_blocks(a)), blocklabels(a))
end

function BlockArrays.blocklengths(a::AbstractGradedUnitRange)
  return labelled.(blocklengths(unlabel_blocks(a)), blocklabels(a))
end

function gradedunitrange_first(a::AbstractUnitRange)
  return labelled(first(unlabel_blocks(a)), label(a[Block(1)]))
end
function Base.first(a::AbstractGradedUnitRange)
  return gradedunitrange_first(a)
end
function Base.first(a::GradedOneTo)
  return gradedunitrange_first(a)
end

Base.iterate(a::AbstractGradedUnitRange) = isempty(a) ? nothing : (first(a), first(a))
function Base.iterate(a::AbstractGradedUnitRange, i)
  i == last(a) && return nothing
  next = a[i + step(a)]
  return (next, next)
end

function firstblockindices(a::AbstractGradedUnitRange)
  return labelled.(firstblockindices(unlabel_blocks(a)), blocklabels(a))
end

function blockedunitrange_getindex(a::AbstractGradedUnitRange, index)
  # This uses `blocklasts` since that is what is stored
  # in `BlockedUnitRange`, maybe abstract that away.
  return labelled(unlabel_blocks(a)[index], get_label(a, index))
end

# The block labels of the corresponding slice.
function blocklabels(a::AbstractUnitRange, indices)
  return map(_blocks(a, indices)) do block
    return label(a[block])
  end
end

function blockedunitrange_getindices(
  ga::AbstractGradedUnitRange, indices::AbstractUnitRange{<:Integer}
)
  a_indices = blockedunitrange_getindices(unlabel_blocks(ga), indices)
  return labelled_blocks(a_indices, blocklabels(ga, indices))
end

# Fixes ambiguity error with:
# ```julia
# blockedunitrange_getindices(::GradedUnitRange, ::AbstractUnitRange{<:Integer})
# ```
# TODO: Try removing once GradedAxes is rewritten for BlockArrays v1.
function blockedunitrange_getindices(a::AbstractGradedUnitRange, indices::BlockSlice)
  return a[indices.block]
end

function blockedunitrange_getindices(ga::AbstractGradedUnitRange, indices::BlockRange)
  return labelled_blocks(unlabel_blocks(ga)[indices], blocklabels(ga, indices))
end

function blockedunitrange_getindices(a::AbstractGradedUnitRange, indices::BlockIndex{1})
  return a[block(indices)][blockindex(indices)]
end

function Base.getindex(a::AbstractGradedUnitRange, index::Integer)
  # This uses `blocklasts` since that is what is stored
  # in `BlockedUnitRange`, maybe abstract that away.
  return labelled(unlabel_blocks(a)[index], get_label(a, index))
end

function Base.getindex(a::AbstractGradedUnitRange, index::Block{1})
  return blockedunitrange_getindex(a, index)
end

function Base.getindex(a::AbstractGradedUnitRange, indices::BlockIndexRange)
  return blockedunitrange_getindices(a, indices)
end

function Base.getindex(
  a::AbstractGradedUnitRange, indices::BlockRange{1,<:Tuple{AbstractUnitRange{Int}}}
)
  return blockedunitrange_getindices(a, indices)
end

# Fixes ambiguity error with `BlockArrays`.
function Base.getindex(
  a::AbstractGradedUnitRange, indices::BlockRange{1,Tuple{Base.OneTo{Int}}}
)
  return blockedunitrange_getindices(a, indices)
end

function Base.getindex(a::AbstractGradedUnitRange, indices::BlockIndex{1})
  return blockedunitrange_getindices(a, indices)
end

# Fixes ambiguity issues with:
# ```julia
# getindex(::BlockedUnitRange, ::BlockSlice)
# getindex(::GradedUnitRange, ::AbstractUnitRange{<:Integer})
# getindex(::GradedUnitRange, ::Any)
# getindex(::AbstractUnitRange, ::AbstractUnitRange{<:Integer})
# ```
# TODO: Maybe not needed once GradedAxes is rewritten
# for BlockArrays v1.
function Base.getindex(a::AbstractGradedUnitRange, indices::BlockSlice)
  return blockedunitrange_getindices(a, indices)
end

function Base.getindex(a::AbstractGradedUnitRange, indices)
  return blockedunitrange_getindices(a, indices)
end

function Base.getindex(a::AbstractGradedUnitRange, indices::AbstractUnitRange{<:Integer})
  return blockedunitrange_getindices(a, indices)
end

# This fixes an issue that `combine_blockaxes` was promoting
# the element type of the axes to `Integer` in broadcasting operations
# that mixed dense and graded axes.
# TODO: Maybe come up with a more general solution.
function BlockArrays.combine_blockaxes(
  a1::GradedOneTo{<:LabelledInteger{T}}, a2::Base.OneTo{T}
) where {T<:Integer}
  combined_blocklasts = sort!(union(unlabel.(blocklasts(a1)), blocklasts(a2)))
  return BlockedOneTo(combined_blocklasts)
end
function BlockArrays.combine_blockaxes(
  a1::Base.OneTo{T}, a2::GradedOneTo{<:LabelledInteger{T}}
) where {T<:Integer}
  return BlockArrays.combine_blockaxes(a2, a1)
end
