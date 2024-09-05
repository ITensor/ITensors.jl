struct BlockedUnitRangeDual{T<:Integer,NondualUnitRange<:AbstractUnitRange} <:
       AbstractBlockedUnitRange{T,Vector{T}}
  nondual_unitrange::NondualUnitRange
end
BlockedUnitRangeDual(a::AbstractUnitRange) = BlockedUnitRangeDual{eltype(a),typeof(a)}(a)

dual(a::AbstractUnitRange) = BlockedUnitRangeDual(a)
nondual(a::BlockedUnitRangeDual) = a.nondual_unitrange
dual(a::BlockedUnitRangeDual) = nondual(a)
flip(a::BlockedUnitRangeDual) = dual(flip(nondual(a)))
nondual(a::AbstractUnitRange) = a
isdual(::AbstractGradedUnitRange) = false
isdual(::BlockedUnitRangeDual) = true
## TODO: Define this to instantiate a dual unit range.
## materialize_dual(a::BlockedUnitRangeDual) = materialize_dual(nondual(a))

Base.first(a::BlockedUnitRangeDual) = label_dual(first(nondual(a)))
Base.last(a::BlockedUnitRangeDual) = label_dual(last(nondual(a)))
Base.step(a::BlockedUnitRangeDual) = label_dual(step(nondual(a)))

Base.view(a::BlockedUnitRangeDual, index::Block{1}) = a[index]

function Base.show(io::IO, a::BlockedUnitRangeDual)
  return print(io, BlockedUnitRangeDual, "(", blocklasts(a), ")")
end

function Base.show(io::IO, mimetype::MIME"text/plain", a::BlockedUnitRangeDual)
  return Base.invoke(
    show, Tuple{typeof(io),MIME"text/plain",AbstractArray}, io, mimetype, a
  )
end

function Base.getindex(a::BlockedUnitRangeDual, indices::AbstractUnitRange{<:Integer})
  return dual(getindex(nondual(a), indices))
end

using BlockArrays: Block, BlockIndexRange, BlockRange

function Base.getindex(a::BlockedUnitRangeDual, indices::Integer)
  return label_dual(getindex(nondual(a), indices))
end

# TODO: Use `label_dual.` here, make broadcasting work?
function Base.getindex(a::BlockedUnitRangeDual, indices::Block{1})
  return dual(getindex(nondual(a), indices))
end

# TODO: Use `label_dual.` here, make broadcasting work?
function Base.getindex(a::BlockedUnitRangeDual, indices::BlockRange)
  return dual(getindex(nondual(a), indices))
end

# fix ambiguity
function Base.getindex(
  a::BlockedUnitRangeDual, indices::BlockRange{1,<:Tuple{AbstractUnitRange{Int}}}
)
  return dual(getindex(nondual(a), indices))
end

function BlockArrays.blocklengths(a::BlockedUnitRangeDual)
  return dual.(blocklengths(nondual(a)))
end

# TODO: Use `label_dual.` here, make broadcasting work?
function unitrangedual_getindices_blocks(a, indices)
  a_indices = getindex(nondual(a), indices)
  return mortar([dual(b) for b in blocks(a_indices)])
end

# TODO: Move this to a `BlockArraysExtensions` library.
function blockedunitrange_getindices(a::BlockedUnitRangeDual, indices::Block{1})
  return a[indices]
end

function Base.getindex(a::BlockedUnitRangeDual, indices::Vector{<:Block{1}})
  return unitrangedual_getindices_blocks(a, indices)
end

function Base.getindex(a::BlockedUnitRangeDual, indices::Vector{<:BlockIndexRange{1}})
  return unitrangedual_getindices_blocks(a, indices)
end

function to_blockindices(a::BlockedUnitRangeDual, indices::UnitRange{<:Integer})
  return to_blockindices(nondual(a), indices)
end

Base.axes(a::BlockedUnitRangeDual) = axes(nondual(a))

using BlockArrays: BlockArrays, Block, BlockSlice
using NDTensors.LabelledNumbers: LabelledUnitRange
function BlockArrays.BlockSlice(b::Block, a::LabelledUnitRange)
  return BlockSlice(b, unlabel(a))
end

using BlockArrays: BlockArrays, BlockSlice
using NDTensors.GradedAxes: BlockedUnitRangeDual, dual
function BlockArrays.BlockSlice(b::Block, r::BlockedUnitRangeDual)
  return BlockSlice(b, dual(r))
end

using NDTensors.LabelledNumbers: LabelledNumbers, label
LabelledNumbers.label(a::BlockedUnitRangeDual) = dual(label(nondual(a)))

using NDTensors.LabelledNumbers: LabelledUnitRange
# The Base version of `length(::AbstractUnitRange)` drops the label.
function Base.length(a::BlockedUnitRangeDual{<:Any,<:LabelledUnitRange})
  return dual(length(nondual(a)))
end
function Base.iterate(a::BlockedUnitRangeDual, i)
  i == last(a) && return nothing
  return dual.(iterate(nondual(a), i))
end
# TODO: Is this a good definition?
Base.unitrange(a::BlockedUnitRangeDual{<:Any,<:AbstractUnitRange}) = a

using NDTensors.LabelledNumbers: LabelledInteger, label, labelled, unlabel
dual(i::LabelledInteger) = labelled(unlabel(i), dual(label(i)))

using BlockArrays: BlockArrays, blockaxes, blocklasts, combine_blockaxes, findblock
BlockArrays.blockaxes(a::BlockedUnitRangeDual) = blockaxes(nondual(a))
BlockArrays.blockfirsts(a::BlockedUnitRangeDual) = label_dual.(blockfirsts(nondual(a)))
BlockArrays.blocklasts(a::BlockedUnitRangeDual) = label_dual.(blocklasts(nondual(a)))
function BlockArrays.findblock(a::BlockedUnitRangeDual, index::Integer)
  return findblock(nondual(a), index)
end

blocklabels(a::BlockedUnitRangeDual) = dual.(blocklabels(nondual(a)))

gradedisequal(::BlockedUnitRangeDual, ::AbstractGradedUnitRange) = false
gradedisequal(::AbstractGradedUnitRange, ::BlockedUnitRangeDual) = false
function gradedisequal(a1::BlockedUnitRangeDual, a2::BlockedUnitRangeDual)
  return gradedisequal(nondual(a1), nondual(a2))
end
function BlockArrays.combine_blockaxes(a1::BlockedUnitRangeDual, a2::BlockedUnitRangeDual)
  return dual(combine_blockaxes(dual(a1), dual(a2)))
end

# This is needed when constructing `CartesianIndices` from
# a tuple of unit ranges that have this kind of dual unit range.
# TODO: See if we can find some more elegant way of constructing
# `CartesianIndices`, maybe by defining conversion of `LabelledInteger`
# to `Int`, defining a more general `convert` function, etc.
function Base.OrdinalRange{Int,Int}(
  r::BlockedUnitRangeDual{<:LabelledInteger{Int},<:LabelledUnitRange{Int,UnitRange{Int}}}
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
function Base.UnitRange{T}(a::BlockedUnitRangeDual{<:LabelledInteger{T}}) where {T<:Integer}
  return UnitRange{T}(nondual(a))
end
