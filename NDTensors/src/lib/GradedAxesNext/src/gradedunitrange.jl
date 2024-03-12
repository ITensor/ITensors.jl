using BlockArrays:
  BlockArrays,
  Block,
  BlockedUnitRange,
  blockedrange,
  blockfirsts,
  blocklasts,
  blocklengths,
  findblock
using ..LabelledNumbers: LabelledNumbers, LabelledInteger, label, labelled, unlabel

const GradedUnitRange{BlockLasts<:Vector{<:LabelledInteger}} = BlockedUnitRange{BlockLasts}

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

function blocklabels(a::GradedUnitRange)
  # Using `blocklasts` here since that is what is stored
  # inside of `BlockedUnitRange`, maybe change that.
  return label.(blocklasts(a))
end

# TODO: This relies on internals of `BlockArrays`, maybe redesign
# to try to avoid that.
# TODO: Define `set_grades`, `set_sector_labels`, `set_labels`.
function unlabel_blocks(a::GradedUnitRange)
  return BlockArrays._BlockedUnitRange(a.first, unlabel.(blocklasts(a)))
end

function BlockArrays.blocklengths(a::GradedUnitRange)
  return labelled.(blocklengths(unlabel_blocks(a)), blocklabels(a))
end

function Base.first(a::GradedUnitRange)
  return labelled(first(unlabel_blocks(a)), label(a, Block(1)))
end

function BlockArrays.blockfirsts(a::GradedUnitRange)
  return labelled.(blockfirsts(unlabel_blocks(a)), blocklabels(a))
end

function LabelledNumbers.label(a::GradedUnitRange, index::Block{1})
  return label(blocklasts(a)[Int(index)])
end

function LabelledNumbers.label(a::GradedUnitRange, index::Integer)
  return label(a, findblock(a, index))
end

# TODO: Define `blocklabels` as `Vector` of `label` for each block.

function gradedunitrange_getindex(a, index)
  return labelled(unlabel_blocks(a)[index], label(a, index))
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
