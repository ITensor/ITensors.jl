using BlockArrays:
  BlockArrays, Block, BlockedUnitRange, blockedrange, blocklasts, blocklengths, findblock
using ..LabelledNumbers: LabelledNumbers, LabelledInteger, label, labelled, unlabel

const GradedUnitRange{BlockLasts<:AbstractVector{<:LabelledInteger}} = BlockedUnitRange{
  BlockLasts
}

function BlockArrays.blockedrange(lblocklengths::AbstractVector{<:LabelledInteger})
  brange = blockedrange(unlabel.(lblocklengths))
  lblocklasts = labelled.(blocklasts(brange), label.(lblocklengths))
  # TODO: `first` is forced to be `Int` in `BlockArrays.BlockedUnitRange`,
  # so this doesn't do anything right now. Make a PR to generalize it.
  firstlength = first(lblocklengths)
  lfirst = oneunit(firstlength)
  return BlockArrays._BlockedUnitRange(lfirst, lblocklasts)
end

function BlockArrays.blockedrange(lblocklengths::AbstractVector{<:Pair{<:Any,<:Integer}})
  return blockedrange(labelled.(last.(lblocklengths), first.(lblocklengths)))
end

# TODO: Call this `ungrade`, `unlabel_blocks`?
# Also, define `set_grades`, `set_sector_labels`, `set_labels`.
function LabelledNumbers.unlabel(a::GradedUnitRange)
  # More general, but relies on internals:
  # return BlockArrays._BlockedUnitRange(unlabel(first(a)), unlabel.(blocklasts(a)))
  @assert isone(first(a))
  return blockedrange(unlabel.(blocklengths(a)))
end

function LabelledNumbers.label(a::GradedUnitRange, index::Block{1})
  return label(blocklasts(a)[Int(index)])
end

function LabelledNumbers.label(a::GradedUnitRange, index::Integer)
  return label(a, findblock(a, index))
end

# TODO: Define `blocklabels` as `Vector` of `label` for each block.

function gradedunitrange_getindex(a, index)
  return labelled(unlabel(a)[index], label(a, index))
end

function Base.getindex(a::GradedUnitRange, index::Integer)
  return gradedunitrange_getindex(a, index)
end

function Base.getindex(a::GradedUnitRange, index::Block{1})
  return gradedunitrange_getindex(a, index)
end

# TODO: Need to add back labels, block structure.
function Base.getindex(a::GradedUnitRange, indices)
  return unlabel(a)[indices]
  return error("Not implemented")
end
