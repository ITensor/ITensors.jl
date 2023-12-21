using BlockArrays:
  BlockArrays, Block, blockfirsts, blocklasts, blocklength, blocklengths, blocks
using EllipsisNotation: Ellipsis, var".."
using TupleTools: TupleTools

value(::Val{N}) where {N} = N

_flatten_tuples(t::Tuple) = t
function _flatten_tuples(t1::Tuple, t2::Tuple, trest::Tuple...)
  return _flatten_tuples((t1..., t2...), trest...)
end
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

struct BlockedPermutation{BlockLength,Length,Blocks<:TupleOfTuples{BlockLength}}
  blocks::Blocks
  global function _BlockedPermutation(blocks::TupleOfTuples)
    len = sum(length, blocks)
    blocklength = length(blocks)
    return new{blocklength,len,typeof(blocks)}(blocks)
  end
end

BlockArrays.blocks(blockedperm::BlockedPermutation) = getfield(blockedperm, :blocks)

function Base.Tuple(blockedperm::BlockedPermutation)
  return flatten_tuples(blocks(blockedperm))
end

function BlockArrays.blocklengths(blockedperm::BlockedPermutation)
  return length.(blocks(blockedperm))
end

function BlockArrays.blockfirsts(blockedperm::BlockedPermutation)
  return _blockfirsts(blocklengths(blockedperm))
end

function BlockArrays.blocklasts(blockedperm::BlockedPermutation)
  return _blocklasts(blocklengths(blockedperm))
end

Base.iterate(permblocks::BlockedPermutation) = iterate(Tuple(permblocks))
Base.iterate(permblocks::BlockedPermutation, state) = iterate(Tuple(permblocks), state)

# blockedperm((4, 3), (2, 1))
function blockedperm(permblocks::Tuple{Vararg{Int}}...; length::Union{Val,Nothing}=nothing)
  return blockedperm(length, permblocks...)
end

function blockedperm(length::Nothing, permblocks::Tuple{Vararg{Int}}...)
  return blockedperm(Val(sum(Base.length, permblocks)), permblocks...)
end

function blockedperm(length::Val, permblocks::Tuple{Vararg{Int}}...)
  @assert value(length) == sum(Base.length, permblocks)
  blockedperm = _BlockedPermutation(permblocks)
  @assert isperm(blockedperm)
  return blockedperm
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

# Block a permutation based on the specified lengths.
# blockperm((4, 3, 2, 1), (2, 2)) == blockedperm((4, 3), (2, 1))
# TODO: Optimize with StaticNumbers.jl or generated functions, see:
# https://discourse.julialang.org/t/avoiding-type-instability-when-slicing-a-tuple/38567
function blockperm(perm::Tuple{Vararg{Int}}, blocklengths::Tuple{Vararg{Int}})
  starts = _blockfirsts(blocklengths)
  stops = _blocklasts(blocklengths)
  return blockedperm(ntuple(i -> perm[starts[i]:stops[i]], length(blocklengths))...)
end

function Base.invperm(blockedperm::BlockedPermutation)
  return blockperm(invperm(Tuple(blockedperm)), blocklengths(blockedperm))
end

Base.length(blockedperm::BlockedPermutation) = length(Tuple(blockedperm))
BlockArrays.blocklength(blockedperm::BlockedPermutation) = length(blocks(blockedperm))

function Base.getindex(blockedperm::BlockedPermutation, i::Int)
  return Tuple(blockedperm)[i]
end

function Base.getindex(blockedperm::BlockedPermutation, I::AbstractUnitRange)
  perm = Tuple(blockedperm)
  return [perm[i] for i in I]
end

function Base.getindex(blockedperm::BlockedPermutation, b::Block)
  return blocks(blockedperm)[Int(b)]
end

# Like `BlockRange`.
function blockeachindex(blockedperm::BlockedPermutation)
  return ntuple(i -> Block(i), blocklength(blockedperm))
end

# Bipartition a vector according to the
# bipartitioned permutation.
# Like `Base.permute!` block out-of-place and blocked.
function blockpermute(v, blockedperm::BlockedPermutation)
  return map(blockperm -> map(i -> v[i], blockperm), blocks(blockedperm))
end

# Version of `indexin` that outputs a `blockedperm`.
function blockedperm_indexin(collection, subs...)
  return blockedperm(map(sub -> BaseExtensions.indexin(sub, collection), subs)...)
end
