struct GradedUnitRangeDual{
  T<:LabelledInteger,NondualUnitRange<:AbstractGradedUnitRange{T}
} <: AbstractGradedUnitRange{T,Vector{T}}
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

function Base.show(io::IO, a::GradedUnitRangeDual)
  return print(io, GradedUnitRangeDual, "(", blocklasts(a), ")")
end

function Base.show(io::IO, mimetype::MIME"text/plain", a::GradedUnitRangeDual)
  return Base.invoke(
    show, Tuple{typeof(io),MIME"text/plain",AbstractArray}, io, mimetype, a
  )
end

function Base.getindex(a::GradedUnitRangeDual, indices::AbstractUnitRange{<:Integer})
  return dual(getindex(nondual(a), indices))
end

using BlockArrays: Block, BlockIndexRange, BlockRange

function Base.getindex(a::GradedUnitRangeDual, indices::Integer)
  return label_dual(getindex(nondual(a), indices))
end

function Base.getindex(a::GradedUnitRangeDual, indices::Block{1})
  return label_dual(getindex(nondual(a), indices))
end

function Base.getindex(a::GradedUnitRangeDual, indices::BlockRange)
  return label_dual(getindex(nondual(a), indices))
end

# fix ambiguity
function Base.getindex(
  a::GradedUnitRangeDual, indices::BlockRange{1,<:Tuple{AbstractUnitRange{Int}}}
)
  return dual(getindex(nondual(a), indices))
end

function BlockArrays.blocklengths(a::GradedUnitRangeDual)
  return dual.(blocklengths(nondual(a)))
end

function unitrangedual_getindices_blocks(a::GradedUnitRangeDual, indices)
  a_indices = getindex(nondual(a), indices)
  return mortar([label_dual(b) for b in blocks(a_indices)])
end

# TODO: Move this to a `BlockArraysExtensions` library.
function blockedunitrange_getindices(a::GradedUnitRangeDual, indices::Block{1})
  return a[indices]
end

function Base.getindex(a::GradedUnitRangeDual, indices::Vector{<:Block{1}})
  return unitrangedual_getindices_blocks(a, indices)
end

function Base.getindex(a::GradedUnitRangeDual, indices::Vector{<:BlockIndexRange{1}})
  return unitrangedual_getindices_blocks(a, indices)
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

using NDTensors.LabelledNumbers: LabelledNumbers, label
LabelledNumbers.label(a::GradedUnitRangeDual) = dual(label(nondual(a)))

using NDTensors.LabelledNumbers: LabelledUnitRange
# The Base version of `length(::AbstractUnitRange)` drops the label.
function Base.length(a::GradedUnitRangeDual{<:Any,<:LabelledUnitRange})
  return dual(length(nondual(a)))
end
function Base.iterate(a::GradedUnitRangeDual, i)
  i == last(a) && return nothing
  return dual.(iterate(nondual(a), i))
end
# TODO: Is this a good definition?
Base.unitrange(a::GradedUnitRangeDual) = a

using NDTensors.LabelledNumbers: LabelledInteger, label, labelled, unlabel
dual(i::LabelledInteger) = labelled(unlabel(i), dual(label(i)))

using BlockArrays: BlockArrays, blockaxes, blocklasts, combine_blockaxes, findblock
BlockArrays.blockaxes(a::GradedUnitRangeDual) = blockaxes(nondual(a))
BlockArrays.blockfirsts(a::GradedUnitRangeDual) = label_dual.(blockfirsts(nondual(a)))
BlockArrays.blocklasts(a::GradedUnitRangeDual) = label_dual.(blocklasts(nondual(a)))
function BlockArrays.findblock(a::GradedUnitRangeDual, index::Integer)
  return findblock(nondual(a), index)
end

blocklabels(a::GradedUnitRangeDual) = dual.(blocklabels(nondual(a)))

gradedisequal(::GradedUnitRangeDual, ::AbstractGradedUnitRange) = false
gradedisequal(::AbstractGradedUnitRange, ::GradedUnitRangeDual) = false
function gradedisequal(a1::GradedUnitRangeDual, a2::GradedUnitRangeDual)
  return gradedisequal(nondual(a1), nondual(a2))
end
function BlockArrays.combine_blockaxes(a1::GradedUnitRangeDual, a2::GradedUnitRangeDual)
  return dual(combine_blockaxes(dual(a1), dual(a2)))
end

# This is needed when constructing `CartesianIndices` from
# a tuple of unit ranges that have this kind of dual unit range.
# TODO: See if we can find some more elegant way of constructing
# `CartesianIndices`, maybe by defining conversion of `LabelledInteger`
# to `Int`, defining a more general `convert` function, etc.
function Base.OrdinalRange{Int,Int}(
  r::GradedUnitRangeDual{<:LabelledInteger{Int},<:LabelledUnitRange{Int,UnitRange{Int}}}
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
function Base.UnitRange{T}(a::GradedUnitRangeDual{<:LabelledInteger{T}}) where {T<:Integer}
  return UnitRange{T}(nondual(a))
end

function unlabel_blocks(a::GradedUnitRangeDual)
  return unlabel_blocks(nondual(a))
end
