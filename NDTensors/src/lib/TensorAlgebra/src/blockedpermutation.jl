using BlockArrays:
  BlockArrays, Block, blockfirsts, blocklasts, blocklength, blocklengths, blocks
using EllipsisNotation: Ellipsis, var".."
using TupleTools: TupleTools

value(::Val{N}) where {N} = N

_flatten_tuples(t::Tuple) = t
function _flatten_tuples(t1::Tuple, t2::Tuple, trest::Tuple...)
  return _flatten_tuples((t1..., t2...), trest...)
end
_flatten_tuples() = ()
flatten_tuples(ts::Tuple) = _flatten_tuples(ts...)

_blocklength(blocklengths::Tuple{Vararg{Int}}) = length(blocklengths)
function _blockfirsts(blocklengths::Tuple{Vararg{Int}})
  return ntuple(_blocklength(blocklengths)) do i
    prev_blocklast =
      isone(i) ? zero(eltype(blocklengths)) : _blocklasts(blocklengths)[i - 1]
    return prev_blocklast + 1
  end
end
_blocklasts(blocklengths::Tuple{Vararg{Int}}) = cumsum(blocklengths)

collect_tuple(x) = (x,)
collect_tuple(x::Ellipsis) = x
collect_tuple(t::Tuple) = t

const TupleOfTuples{N} = Tuple{Vararg{Tuple{Vararg{Int}},N}}

abstract type AbstractBlockedPermutation{BlockLength,Length} end

BlockArrays.blocks(blockedperm::AbstractBlockedPermutation) = error("Not implemented")

function Base.Tuple(blockedperm::AbstractBlockedPermutation)
  return flatten_tuples(blocks(blockedperm))
end

function BlockArrays.blocklengths(blockedperm::AbstractBlockedPermutation)
  return length.(blocks(blockedperm))
end

function BlockArrays.blockfirsts(blockedperm::AbstractBlockedPermutation)
  return _blockfirsts(blocklengths(blockedperm))
end

function BlockArrays.blocklasts(blockedperm::AbstractBlockedPermutation)
  return _blocklasts(blocklengths(blockedperm))
end

Base.iterate(permblocks::AbstractBlockedPermutation) = iterate(Tuple(permblocks))
function Base.iterate(permblocks::AbstractBlockedPermutation, state)
  return iterate(Tuple(permblocks), state)
end

# Block a permutation based on the specified lengths.
# blockperm((4, 3, 2, 1), (2, 2)) == blockedperm((4, 3), (2, 1))
# TODO: Optimize with StaticNumbers.jl or generated functions, see:
# https://discourse.julialang.org/t/avoiding-type-instability-when-slicing-a-tuple/38567
function blockperm(perm::Tuple{Vararg{Int}}, blocklengths::Tuple{Vararg{Int}})
  starts = _blockfirsts(blocklengths)
  stops = _blocklasts(blocklengths)
  return blockedperm(ntuple(i -> perm[starts[i]:stops[i]], length(blocklengths))...)
end

function Base.invperm(blockedperm::AbstractBlockedPermutation)
  return blockperm(invperm(Tuple(blockedperm)), blocklengths(blockedperm))
end

Base.length(blockedperm::AbstractBlockedPermutation) = length(Tuple(blockedperm))
function BlockArrays.blocklength(blockedperm::AbstractBlockedPermutation)
  return length(blocks(blockedperm))
end

function Base.getindex(blockedperm::AbstractBlockedPermutation, i::Int)
  return Tuple(blockedperm)[i]
end

function Base.getindex(blockedperm::AbstractBlockedPermutation, I::AbstractUnitRange)
  perm = Tuple(blockedperm)
  return [perm[i] for i in I]
end

function Base.getindex(blockedperm::AbstractBlockedPermutation, b::Block)
  return blocks(blockedperm)[Int(b)]
end

# Like `BlockRange`.
function blockeachindex(blockedperm::AbstractBlockedPermutation)
  return ntuple(i -> Block(i), blocklength(blockedperm))
end

#
# Constructors
#

# Bipartition a vector according to the
# bipartitioned permutation.
# Like `Base.permute!` block out-of-place and blocked.
function blockpermute(v, blockedperm::AbstractBlockedPermutation)
  return map(blockperm -> map(i -> v[i], blockperm), blocks(blockedperm))
end

# blockedperm((4, 3), (2, 1))
function blockedperm(permblocks::Tuple{Vararg{Int}}...; length::Union{Val,Nothing}=nothing)
  return blockedperm(length, permblocks...)
end

function blockedperm(length::Nothing, permblocks::Tuple{Vararg{Int}}...)
  return blockedperm(Val(sum(Base.length, permblocks; init=zero(Bool))), permblocks...)
end

# blockedperm((3, 2), 1) == blockedperm((3, 2), (1,))
function blockedperm(permblocks::Union{Tuple{Vararg{Int}},Int}...; kwargs...)
  return blockedperm(collect_tuple.(permblocks)...; kwargs...)
end

function blockedperm(permblocks::Union{Tuple{Vararg{Int}},Int,Ellipsis}...; kwargs...)
  return blockedperm(collect_tuple.(permblocks)...; kwargs...)
end

function _blockedperm_length(::Nothing, specified_perm::Tuple{Vararg{Int}})
  return maximum(specified_perm)
end

function _blockedperm_length(vallength::Val, specified_perm::Tuple{Vararg{Int}})
  return value(vallength)
end

# blockedperm((4, 3), .., 1) == blockedperm((4, 3), 2, 1)
# blockedperm((4, 3), .., 1; length=Val(5)) == blockedperm((4, 3), 2, 5, 1)
function blockedperm(
  permblocks::Union{Tuple{Vararg{Int}},Ellipsis}...; length::Union{Val,Nothing}=nothing
)
  # Check there is only one `Ellipsis`.
  @assert isone(count(x -> x isa Ellipsis, permblocks))
  specified_permblocks = filter(x -> !(x isa Ellipsis), permblocks)
  unspecified_dim = findfirst(x -> x isa Ellipsis, permblocks)
  specified_perm = flatten_tuples(specified_permblocks)
  len = _blockedperm_length(length, specified_perm)
  unspecified_dims = Tuple(setdiff(Base.OneTo(len), flatten_tuples(specified_permblocks)))
  permblocks_specified = TupleTools.insertat(permblocks, unspecified_dim, unspecified_dims)
  return blockedperm(permblocks_specified...)
end

# Version of `indexin` that outputs a `blockedperm`.
function blockedperm_indexin(collection, subs...)
  return blockedperm(map(sub -> BaseExtensions.indexin(sub, collection), subs)...)
end

struct BlockedPermutation{BlockLength,Length,Blocks<:TupleOfTuples{BlockLength}} <:
       AbstractBlockedPermutation{BlockLength,Length}
  blocks::Blocks
  global function _BlockedPermutation(blocks::TupleOfTuples)
    len = sum(length, blocks; init=zero(Bool))
    blocklength = length(blocks)
    return new{blocklength,len,typeof(blocks)}(blocks)
  end
end

BlockArrays.blocks(blockedperm::BlockedPermutation) = getfield(blockedperm, :blocks)

function blockedperm(length::Val, permblocks::Tuple{Vararg{Int}}...)
  @assert value(length) == sum(Base.length, permblocks; init=zero(Bool))
  blockedperm = _BlockedPermutation(permblocks)
  @assert isperm(blockedperm)
  return blockedperm
end

trivialperm(length::Union{Integer,Val}) = ntuple(identity, length)

struct BlockedTrivialPermutation{BlockLength,Length,Blocks<:TupleOfTuples{BlockLength}} <:
       AbstractBlockedPermutation{BlockLength,Length}
  blocks::Blocks
  global function _BlockedTrivialPermutation(blocklengths::Tuple{Vararg{Int}})
    len = sum(blocklengths; init=zero(Bool))
    blocklength = length(blocklengths)
    permblocks = blocks(blockperm(trivialperm(len), blocklengths))
    return new{blocklength,len,typeof(permblocks)}(permblocks)
  end
end

BlockArrays.blocks(blockedperm::BlockedTrivialPermutation) = getfield(blockedperm, :blocks)

function blockedtrivialperm(blocklengths::Tuple{Vararg{Int}})
  return _BlockedTrivialPermutation(blocklengths)
end

function trivialperm(blockedperm::AbstractBlockedPermutation)
  return blockedtrivialperm(blocklengths(blockedperm))
end
Base.invperm(blockedperm::BlockedTrivialPermutation) = blockedperm
