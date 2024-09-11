struct UnitRangeDual{T<:Integer,NondualUnitRange<:AbstractUnitRange} <: AbstractUnitRange{T}
  nondual_unitrange::NondualUnitRange
end
UnitRangeDual(a::AbstractUnitRange) = UnitRangeDual{eltype(a),typeof(a)}(a)

dual(a::AbstractUnitRange) = UnitRangeDual(a)
nondual(a::UnitRangeDual) = a.nondual_unitrange
dual(a::UnitRangeDual) = nondual(a)
flip(a::UnitRangeDual) = dual(flip(nondual(a)))
nondual(a::AbstractUnitRange) = a
isdual(::UnitRangeDual) = true
## TODO: Define this to instantiate a dual unit range.
## materialize_dual(a::UnitRangeDual) = materialize_dual(nondual(a))

Base.first(a::UnitRangeDual) = first(nondual(a))
Base.last(a::UnitRangeDual) = last(nondual(a))
Base.step(a::UnitRangeDual) = step(nondual(a))

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
function Base.getindex(a::UnitRangeDual, indices::Block{1})
  return dual(getindex(nondual(a), indices))
end

# TODO: Use `label_dual.` here, make broadcasting work?
function Base.getindex(a::UnitRangeDual, indices::BlockRange)
  return dual(getindex(nondual(a), indices))
end

# fix ambiguity
function Base.getindex(
  a::UnitRangeDual, indices::BlockRange{1,<:Tuple{AbstractUnitRange{Int}}}
)
  return dual(getindex(nondual(a), indices))
end

function BlockArrays.blocklengths(a::UnitRangeDual)
  return dual.(blocklengths(nondual(a)))
end

function unitrangedual_getindices_blocks(a, indices)
  a_indices = blockedunitrange_getindices(nondual(a), indices)
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

using BlockArrays: BlockArrays, BlockSlice
using NDTensors.GradedAxes: UnitRangeDual, dual
function BlockArrays.BlockSlice(b::Block, r::UnitRangeDual)
  return BlockSlice(b, dual(r))
end

function Base.iterate(a::UnitRangeDual, i)
  i == last(a) && return nothing
  return dual.(iterate(nondual(a), i))
end
# TODO: Is this a good definition?
Base.unitrange(a::UnitRangeDual{<:Any,<:AbstractUnitRange}) = a

using BlockArrays: BlockArrays, blockaxes, blocklasts, combine_blockaxes, findblock
BlockArrays.blockaxes(a::UnitRangeDual) = blockaxes(nondual(a))
BlockArrays.blockfirsts(a::UnitRangeDual) = blockfirsts(nondual(a))
BlockArrays.blocklasts(a::UnitRangeDual) = blocklasts(nondual(a))
function BlockArrays.findblock(a::UnitRangeDual, index::Integer)
  return findblock(nondual(a), index)
end

gradedisequal(::UnitRangeDual, ::AbstractGradedUnitRange) = false
gradedisequal(::AbstractGradedUnitRange, ::UnitRangeDual) = false
function gradedisequal(a1::UnitRangeDual, a2::UnitRangeDual)
  return gradedisequal(nondual(a1), nondual(a2))
end
function BlockArrays.combine_blockaxes(a1::UnitRangeDual, a2::UnitRangeDual)
  return dual(combine_blockaxes(dual(a1), dual(a2)))
end
