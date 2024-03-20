struct UnitRangeDual{T,NondualUnitRange<:AbstractUnitRange} <: AbstractUnitRange{T}
  nondual_unitrange::NondualUnitRange
end
UnitRangeDual(a::AbstractUnitRange) = UnitRangeDual{eltype(a),typeof(a)}(a)

dual(a::AbstractUnitRange) = UnitRangeDual(a)
nondual(a::UnitRangeDual) = a.nondual_unitrange
dual(a::UnitRangeDual) = nondual(a)
nondual(a::AbstractUnitRange) = a
## TODO: Define this to instantiate a dual unit range.
## materialize_dual(a::UnitRangeDual) = materialize_dual(nondual(a))

Base.first(a::UnitRangeDual) = label_dual(first(nondual(a)))
Base.last(a::UnitRangeDual) = label_dual(last(nondual(a)))
Base.step(a::UnitRangeDual) = label_dual(step(nondual(a)))

function Base.getindex(a::UnitRangeDual, indices::AbstractUnitRange{<:Integer})
  return label_dual.(getindex(nondual(a), indices))
end

using BlockArrays: Block, BlockIndexRange, BlockRange

function Base.getindex(a::UnitRangeDual, indices::Integer)
  return label_dual(getindex(nondual(a), indices))
end

# TODO: Use `label_dual.` here, make broadcasting work?
Base.getindex(a::UnitRangeDual, indices::Block{1}) = dual(getindex(nondual(a), indices))

# TODO: Use `label_dual.` here, make broadcasting work?
Base.getindex(a::UnitRangeDual, indices::BlockRange) = dual(getindex(nondual(a), indices))

# TODO: Fix this, actually take the `dual`.
# TODO: Use `label_dual.` here, make broadcasting work?
Base.getindex(a::UnitRangeDual, indices::Vector{<:Block{1}}) = getindex(nondual(a), indices)

function Base.getindex(a::UnitRangeDual, indices::Vector{<:BlockIndexRange{1}})
  return label_dual.(getindex(nondual(a), indices))
end

# Fix ambiguity errors with Base.
## for type in [
##   :Integer,
##   :(AbstractUnitRange{<:Integer}),
##   :(Block{1}),
##   :BlockRange,
## ]
##   @eval Base.getindex(a::UnitRangeDual, indices::$type) = dual(getindex(nondual(a), indices))
## end

Base.axes(a::UnitRangeDual) = axes(nondual(a))

using BlockArrays: BlockArrays, Block, BlockSlice
using NDTensors.LabelledNumbers: LabelledUnitRange
function BlockArrays.BlockSlice(b::Block, a::LabelledUnitRange)
  return BlockSlice(b, unlabel(a))
end

using BlockArrays: BlockArrays, blockaxes, blocklasts, findblock
BlockArrays.blockaxes(a::UnitRangeDual) = blockaxes(nondual(a))
BlockArrays.blockfirsts(a::UnitRangeDual) = label_dual.(blockfirsts(nondual(a)))
BlockArrays.blocklasts(a::UnitRangeDual) = label_dual.(blocklasts(nondual(a)))
BlockArrays.findblock(a::UnitRangeDual, index::Integer) = findblock(nondual(a), index)
