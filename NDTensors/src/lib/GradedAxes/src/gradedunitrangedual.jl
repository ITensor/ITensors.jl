struct GradedUnitRangeDual{
  T,BlockLasts,NondualUnitRange<:AbstractGradedUnitRange{T,BlockLasts}
} <: AbstractGradedUnitRange{T,BlockLasts}
  nondual_unitrange::NondualUnitRange
end

dual(a::AbstractGradedUnitRange) = GradedUnitRangeDual(a)
nondual(a::GradedUnitRangeDual) = a.nondual_unitrange
dual(a::GradedUnitRangeDual) = nondual(a)
flip(a::GradedUnitRangeDual) = dual(flip(nondual(a)))
isdual(::GradedUnitRangeDual) = true
## TODO: Define this to instantiate a dual unit range.
## materialize_dual(a::GradedUnitRangeDual) = materialize_dual(nondual(a))

Base.first(a::GradedUnitRangeDual) = label_dual(first(nondual(a)))
Base.last(a::GradedUnitRangeDual) = label_dual(last(nondual(a)))
Base.step(a::GradedUnitRangeDual) = label_dual(step(nondual(a)))

Base.view(a::GradedUnitRangeDual, index::Block{1}) = a[index]

function blockedunitrange_getindices(
  a::GradedUnitRangeDual, indices::AbstractUnitRange{<:Integer}
)
  return dual(getindex(nondual(a), indices))
end

using BlockArrays: Block, BlockIndexRange, BlockRange

function blockedunitrange_getindices(a::GradedUnitRangeDual, indices::Integer)
  return label_dual(getindex(nondual(a), indices))
end

function blockedunitrange_getindices(a::GradedUnitRangeDual, indices::Block{1})
  return label_dual(getindex(nondual(a), indices))
end

function blockedunitrange_getindices(a::GradedUnitRangeDual, indices::BlockRange)
  return label_dual(getindex(nondual(a), indices))
end

# fix ambiguity
function blockedunitrange_getindices(
  a::GradedUnitRangeDual, indices::BlockRange{1,<:Tuple{AbstractUnitRange{Int}}}
)
  return dual(getindex(nondual(a), indices))
end

function BlockArrays.blocklengths(a::GradedUnitRangeDual)
  return dual.(blocklengths(nondual(a)))
end

function gradedunitrangedual_getindices_blocks(a::GradedUnitRangeDual, indices)
  a_indices = getindex(nondual(a), indices)
  return mortar([label_dual(b) for b in blocks(a_indices)])
end

# TODO: Move this to a `BlockArraysExtensions` library.
function blockedunitrange_getindices(a::GradedUnitRangeDual, indices::Vector{<:Block{1}})
  return gradedunitrangedual_getindices_blocks(a, indices)
end

function blockedunitrange_getindices(
  a::GradedUnitRangeDual, indices::Vector{<:BlockIndexRange{1}}
)
  return gradedunitrangedual_getindices_blocks(a, indices)
end

Base.axes(a::GradedUnitRangeDual) = axes(nondual(a))

using BlockArrays: BlockArrays, Block, BlockSlice
using NDTensors.LabelledNumbers: LabelledUnitRange
function BlockArrays.BlockSlice(b::Block, a::LabelledUnitRange)
  return BlockSlice(b, unlabel(a))
end

using BlockArrays: BlockArrays, BlockSlice
using NDTensors.GradedAxes: GradedUnitRangeDual, dual
function BlockArrays.BlockSlice(b::Block, r::GradedUnitRangeDual)
  return BlockSlice(b, dual(r))
end

using NDTensors.LabelledNumbers: LabelledNumbers, LabelledUnitRange, label
function Base.iterate(a::GradedUnitRangeDual, i)
  i == last(a) && return nothing
  return dual.(iterate(nondual(a), i))
end

using NDTensors.LabelledNumbers: LabelledInteger, label, labelled, unlabel
using BlockArrays: BlockArrays, blockaxes, blocklasts, combine_blockaxes, findblock
BlockArrays.blockaxes(a::GradedUnitRangeDual) = blockaxes(nondual(a))
BlockArrays.blockfirsts(a::GradedUnitRangeDual) = label_dual.(blockfirsts(nondual(a)))
BlockArrays.blocklasts(a::GradedUnitRangeDual) = label_dual.(blocklasts(nondual(a)))
function BlockArrays.findblock(a::GradedUnitRangeDual, index::Integer)
  return findblock(nondual(a), index)
end

blocklabels(a::GradedUnitRangeDual) = dual.(blocklabels(nondual(a)))

function BlockArrays.combine_blockaxes(a1::GradedUnitRangeDual, a2::GradedUnitRangeDual)
  return dual(combine_blockaxes(nondual(a1), nondual(a2)))
end

function unlabel_blocks(a::GradedUnitRangeDual)
  return unlabel_blocks(nondual(a))
end
