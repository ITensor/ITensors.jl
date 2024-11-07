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
  blockisequal,
  blocklength,
  blocklengths,
  findblock,
  findblockindex,
  mortar,
  sortedunion
using Compat: allequal
using FillArrays: Fill
using ..LabelledNumbers:
  LabelledNumbers,
  LabelledInteger,
  LabelledUnitRange,
  label,
  label_type,
  labelled,
  labelled_isequal,
  unlabel

abstract type AbstractGradedUnitRange{T,BlockLasts} <:
              AbstractBlockedUnitRange{T,BlockLasts} end

struct GradedUnitRange{T,BlockLasts<:Vector{T}} <: AbstractGradedUnitRange{T,BlockLasts}
  first::T
  lasts::BlockLasts
end

struct GradedOneTo{T,BlockLasts<:Vector{T}} <: AbstractGradedUnitRange{T,BlockLasts}
  lasts::BlockLasts

  # assume that lasts is sorted, no checks carried out here
  function GradedOneTo(lasts::BlockLasts) where {T<:Integer,BlockLasts<:AbstractVector{T}}
    Base.require_one_based_indexing(lasts)
    isempty(lasts) || first(lasts) >= 0 || throw(ArgumentError("blocklasts must be >= 0"))
    return new{T,BlockLasts}(lasts)
  end
  function GradedOneTo(lasts::BlockLasts) where {T<:Integer,BlockLasts<:Tuple{T,Vararg{T}}}
    first(lasts) >= 0 || throw(ArgumentError("blocklasts must be >= 0"))
    return new{T,BlockLasts}(lasts)
  end
end

function Base.show(io::IO, ::MIME"text/plain", g::AbstractGradedUnitRange)
  v = map(b -> label(b) => unlabel(b), blocks(g))
  println(io, typeof(g))
  return print(io, join(repr.(v), '\n'))
end

function Base.show(io::IO, g::AbstractGradedUnitRange)
  v = map(b -> label(b) => unlabel(b), blocks(g))
  return print(io, nameof(typeof(g)), '[', join(repr.(v), ", "), ']')
end

# == is just a range comparison that ignores labels. Need dedicated function to check equality.
struct NoLabel end
blocklabels(r::AbstractUnitRange) = Fill(NoLabel(), blocklength(r))
blocklabels(la::LabelledUnitRange) = [label(la)]

function LabelledNumbers.labelled_isequal(a1::AbstractUnitRange, a2::AbstractUnitRange)
  return blockisequal(a1, a2) && (blocklabels(a1) == blocklabels(a2))
end

function space_isequal(a1::AbstractUnitRange, a2::AbstractUnitRange)
  return (isdual(a1) == isdual(a2)) && labelled_isequal(a1, a2)
end

# needed in BlockSparseArrays
function Base.AbstractUnitRange{T}(
  a::AbstractGradedUnitRange{<:LabelledInteger{T}}
) where {T}
  return unlabel_blocks(a)
end

# TODO: Use `TypeParameterAccessors`.
Base.eltype(::Type{<:GradedUnitRange{T}}) where {T} = T
LabelledNumbers.label_type(g::AbstractGradedUnitRange) = label_type(typeof(g))
LabelledNumbers.label_type(T::Type{<:AbstractGradedUnitRange}) = label_type(eltype(T))

function gradedrange(lblocklengths::AbstractVector{<:LabelledInteger})
  brange = blockedrange(unlabel.(lblocklengths))
  lblocklasts = labelled.(blocklasts(brange), label.(lblocklengths))
  return GradedOneTo(lblocklasts)
end

# To help with generic code.
function BlockArrays.blockedrange(lblocklengths::AbstractVector{<:LabelledInteger})
  return gradedrange(lblocklengths)
end

Base.last(a::AbstractGradedUnitRange) = isempty(a.lasts) ? first(a) - 1 : last(a.lasts)

function gradedrange(lblocklengths::AbstractVector{<:Pair{<:Any,<:Integer}})
  return gradedrange(labelled.(last.(lblocklengths), first.(lblocklengths)))
end

function labelled_blocks(a::BlockedOneTo, labels)
  # TODO: Use `blocklasts(a)`? That might
  # cause a recursive loop.
  return GradedOneTo(labelled.(a.lasts, labels))
end
function labelled_blocks(a::BlockedUnitRange, labels)
  # TODO: Use `first(a)` and `blocklasts(a)`? Those might
  # cause a recursive loop.
  return GradedUnitRange(labelled(a.first, labels[1]), labelled.(a.lasts, labels))
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
function unlabel_blocks(a::GradedOneTo)
  # TODO: Use `blocklasts(a)`.
  return BlockedOneTo(unlabel.(a.lasts))
end
function unlabel_blocks(a::GradedUnitRange)
  return BlockArrays._BlockedUnitRange(a.first, unlabel.(a.lasts))
end

## BlockedUnitRange interface

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

Base.iterate(a::AbstractGradedUnitRange) = isempty(a) ? nothing : (first(a), first(a))
function Base.iterate(a::AbstractGradedUnitRange, i)
  i == last(a) && return nothing
  next = a[i + step(a)]
  return (next, next)
end

function firstblockindices(a::AbstractGradedUnitRange)
  return labelled.(firstblockindices(unlabel_blocks(a)), blocklabels(a))
end

function blockedunitrange_getindices(a::AbstractGradedUnitRange, index::Block{1})
  return labelled(unlabel_blocks(a)[index], get_label(a, index))
end

function blockedunitrange_getindices(a::AbstractGradedUnitRange, indices::Vector{<:Integer})
  return map(index -> a[index], indices)
end

function blockedunitrange_getindices(
  a::AbstractGradedUnitRange,
  indices::BlockVector{<:BlockIndex{1},<:Vector{<:BlockIndexRange{1}}},
)
  return mortar(map(b -> a[b], blocks(indices)))
end

function blockedunitrange_getindices(a::AbstractGradedUnitRange, index)
  return labelled(unlabel_blocks(a)[index], get_label(a, index))
end

function blockedunitrange_getindices(a::AbstractGradedUnitRange, indices::BlockIndexRange)
  return a[block(indices)][only(indices.indices)]
end

function blockedunitrange_getindices(
  a::AbstractGradedUnitRange, indices::AbstractVector{<:Union{Block{1},BlockIndexRange{1}}}
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
  return labelled(unlabel_blocks(a)[index], get_label(a, index))
end

function Base.getindex(a::AbstractGradedUnitRange, index::Block{1})
  return blockedunitrange_getindices(a, index)
end

function Base.getindex(a::AbstractGradedUnitRange, indices::BlockIndexRange)
  return blockedunitrange_getindices(a, indices)
end

# fix ambiguities
function Base.getindex(
  a::AbstractGradedUnitRange, indices::BlockArrays.BlockRange{1,<:Tuple{Base.OneTo}}
)
  return blockedunitrange_getindices(a, indices)
end
function Base.getindex(
  a::AbstractGradedUnitRange, indices::BlockRange{1,<:Tuple{AbstractUnitRange{Int}}}
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
  a1::AbstractGradedUnitRange{<:LabelledInteger{T}}, a2::AbstractUnitRange{T}
) where {T<:Integer}
  combined_blocklasts = sort!(union(unlabel.(blocklasts(a1)), blocklasts(a2)))
  return BlockedOneTo(combined_blocklasts)
end
function BlockArrays.combine_blockaxes(
  a1::AbstractUnitRange{T}, a2::AbstractGradedUnitRange{<:LabelledInteger{T}}
) where {T<:Integer}
  return BlockArrays.combine_blockaxes(a2, a1)
end

# preserve labels inside combine_blockaxes
function BlockArrays.combine_blockaxes(a::GradedOneTo, b::GradedOneTo)
  return GradedOneTo(sortedunion(blocklasts(a), blocklasts(b)))
end
function BlockArrays.combine_blockaxes(a::GradedUnitRange, b::GradedUnitRange)
  new_blocklasts = sortedunion(blocklasts(a), blocklasts(b))
  new_first = labelled(oneunit(eltype(new_blocklasts)), label(first(new_blocklasts)))
  return GradedUnitRange(new_first, new_blocklasts)
end

# preserve axes in SubArray
Base.axes(S::Base.Slice{<:AbstractGradedUnitRange}) = (S.indices,)

# Version of length that checks that all blocks have the same label
# and returns a labelled length with that label.
function labelled_length(a::AbstractBlockVector{<:Integer})
  blocklabels = label.(blocks(a))
  @assert allequal(blocklabels)
  return labelled(unlabel(length(a)), first(blocklabels))
end

# TODO: Make sure this handles block labels (AbstractGradedUnitRange) correctly.
# TODO: Make a special case for `BlockedVector{<:Block{1},<:BlockRange{1}}`?
# For example:
# ```julia
# blocklengths = map(bs -> sum(b -> length(a[b]), bs), blocks(indices))
# return blockedrange(blocklengths)
# ```
function blockedunitrange_getindices(
  a::AbstractGradedUnitRange, indices::AbstractBlockVector{<:Block{1}}
)
  blks = map(bs -> mortar(map(b -> a[b], bs)), blocks(indices))
  # We pass `length.(blks)` to `mortar` in order
  # to pass block labels to the axes of the output,
  # if they exist. This makes it so that
  # `only(axes(a[indices])) isa `GradedUnitRange`
  # if `a isa `GradedUnitRange`, for example.
  return mortar(blks, labelled_length.(blks))
end
