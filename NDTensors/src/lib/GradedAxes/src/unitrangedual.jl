struct UnitRangeDual{T,NondualUnitRange<:AbstractUnitRange} <: AbstractUnitRange{T}
  nondual_unitrange::NondualUnitRange
end
UnitRangeDual(a::AbstractUnitRange) = UnitRangeDual{eltype(a),typeof(a)}(a)

dual(a::AbstractUnitRange) = UnitRangeDual(a)
nondual(a::UnitRangeDual) = a.nondual_unitrange
dual(a::UnitRangeDual) = nondual(a)
flip(a::UnitRangeDual) = dual(flip(nondual(a)))
nondual(a::AbstractUnitRange) = a
isdual(::AbstractUnitRange) = false
isdual(::UnitRangeDual) = true
## TODO: Define this to instantiate a dual unit range.
## materialize_dual(a::UnitRangeDual) = materialize_dual(nondual(a))

Base.first(a::UnitRangeDual) = label_dual(first(nondual(a)))
Base.last(a::UnitRangeDual) = label_dual(last(nondual(a)))
Base.step(a::UnitRangeDual) = label_dual(step(nondual(a)))

Base.view(a::UnitRangeDual, index::Block{1}) = a[index]

function Base.show(io::IO, a::UnitRangeDual)
  return print(io, UnitRangeDual, "(", blocklasts(a), ")")
end

function Base.show(io::IO, mimetype::MIME"text/plain", a::UnitRangeDual)
  return Base.invoke(
    show, Tuple{typeof(io),MIME"text/plain",AbstractArray}, io, mimetype, a
  )
end

function Base.getindex(a::UnitRangeDual, indices::AbstractUnitRange{<:Integer})
  return dual(getindex(nondual(a), indices))
end

using BlockArrays: Block, BlockIndexRange, BlockRange

function Base.getindex(a::UnitRangeDual, indices::Integer)
  return label_dual(getindex(nondual(a), indices))
end

# TODO: Use `label_dual.` here, make broadcasting work?
Base.getindex(a::UnitRangeDual, indices::Block{1}) = dual(getindex(nondual(a), indices))

# TODO: Use `label_dual.` here, make broadcasting work?
Base.getindex(a::UnitRangeDual, indices::BlockRange) = dual(getindex(nondual(a), indices))

# TODO: Use `label_dual.` here, make broadcasting work?
function unitrangedual_getindices_blocks(a, indices)
  a_indices = getindex(nondual(a), indices)
  return mortar([dual(b) for b in blocks(a_indices)])
end

# TODO: Move this to a `BlockArraysExtensions` library.
function blockedunitrange_getindices(a::UnitRangeDual, indices::Block{1})
  return a[indices]
end

function Base.getindex(a::UnitRangeDual, indices::Vector{<:Block{1}})
  return unitrangedual_getindices_blocks(a, indices)
end

function Base.getindex(a::UnitRangeDual, indices::Vector{<:BlockIndexRange{1}})
  return unitrangedual_getindices_blocks(a, indices)
end

function to_blockindices(a::UnitRangeDual, indices::UnitRange{<:Integer})
  return to_blockindices(nondual(a), indices)
end

Base.axes(a::UnitRangeDual) = axes(nondual(a))

using BlockArrays: BlockArrays, Block, BlockSlice
using NDTensors.LabelledNumbers: LabelledUnitRange
function BlockArrays.BlockSlice(b::Block, a::LabelledUnitRange)
  return BlockSlice(b, unlabel(a))
end

using BlockArrays: BlockArrays, BlockSlice
using NDTensors.GradedAxes: UnitRangeDual, dual
function BlockArrays.BlockSlice(b::Block, r::UnitRangeDual)
  return BlockSlice(b, dual(r))
end

using NDTensors.LabelledNumbers: LabelledNumbers, label
LabelledNumbers.label(a::UnitRangeDual) = dual(label(nondual(a)))

using NDTensors.LabelledNumbers: LabelledUnitRange
# The Base version of `length(::AbstractUnitRange)` drops the label.
function Base.length(a::UnitRangeDual{<:Any,<:LabelledUnitRange})
  return dual(length(nondual(a)))
end
function Base.iterate(a::UnitRangeDual, i)
  i == last(a) && return nothing
  return dual.(iterate(nondual(a), i))
end
# TODO: Is this a good definition?
Base.unitrange(a::UnitRangeDual{<:Any,<:AbstractUnitRange}) = a

using NDTensors.LabelledNumbers: LabelledInteger, label, labelled, unlabel
dual(i::LabelledInteger) = labelled(unlabel(i), dual(label(i)))

using BlockArrays: BlockArrays, blockaxes, blocklasts, combine_blockaxes, findblock
BlockArrays.blockaxes(a::UnitRangeDual) = blockaxes(nondual(a))
BlockArrays.blockfirsts(a::UnitRangeDual) = label_dual.(blockfirsts(nondual(a)))
BlockArrays.blocklasts(a::UnitRangeDual) = label_dual.(blocklasts(nondual(a)))
BlockArrays.findblock(a::UnitRangeDual, index::Integer) = findblock(nondual(a), index)

blocklabels(a::UnitRangeDual) = dual.(blocklabels(nondual(a)))

function BlockArrays.combine_blockaxes(a1::UnitRangeDual, a2::UnitRangeDual)
  return dual(combine_blockaxes(dual(a1), dual(a2)))
end

# This is needed when constructing `CartesianIndices` from
# a tuple of unit ranges that have this kind of dual unit range.
# TODO: See if we can find some more elegant way of constructing
# `CartesianIndices`, maybe by defining conversion of `LabelledInteger`
# to `Int`, defining a more general `convert` function, etc.
function Base.OrdinalRange{Int,Int}(
  r::UnitRangeDual{<:LabelledInteger{Int},<:LabelledUnitRange{Int,UnitRange{Int}}}
)
  # TODO: Implement this broadcasting operation and use it here.
  # return Int.(r)
  return unlabel(nondual(r))
end

# This is only needed in certain Julia versions below 1.10
# (for example Julia 1.6).
# TODO: Delete this once we drop Julia 1.6 support.
# The type constraint `T<:Integer` is needed to avoid an ambiguity
# error with a conversion method in Base.
function Base.UnitRange{T}(a::UnitRangeDual{<:LabelledInteger{T}}) where {T<:Integer}
  return UnitRange{T}(nondual(a))
end
