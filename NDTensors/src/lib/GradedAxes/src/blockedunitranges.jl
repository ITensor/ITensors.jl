module BlockedUnitRanges
using Base: @propagate_inbounds
using BlockArrays:
  BlockArrays,
  Block,
  BlockIndex,
  BlockIndexRange,
  BlockRange,
  BlockSlice,
  RangeCumsum,
  blockaxes,
  blockfirsts,
  blocklasts,
  blocklength,
  blocklengths,
  blocksize
using FillArrays: Fill

# Essentially the same as `BlockArrays.BlockedUnitRange`
# but allows for a more general element type.
struct BlockedUnitRange{T,CS} <: AbstractUnitRange{T}
  first::T
  lasts::CS
  global function _BlockedUnitRange(f, cs)
    @assert typeof(f) == eltype(cs)
    return new{typeof(f),typeof(cs)}(f, cs)
  end
end

@inline _BlockedUnitRange(cs) = _BlockedUnitRange(one(first(cs)), cs)

BlockedUnitRange(::BlockedUnitRange) = throw(ArgumentError("Forbidden due to ambiguity"))
_blocklengths2blocklasts(blocks) = eltype(blocks).(cumsum(blocks)) # extra level to allow changing default cumsum behaviour
@inline blockedrange(blocks::Union{Tuple,AbstractVector}) =
  _BlockedUnitRange(_blocklengths2blocklasts(blocks))

@inline BlockArrays.blockfirsts(a::BlockedUnitRange) =
  [a.first; @views(a.lasts[1:(end - 1)]) .+ 1]
# optimize common cases
@inline function BlockArrays.blockfirsts(
  a::BlockedUnitRange{<:Union{Vector,RangeCumsum{<:Any,<:UnitRange}}}
)
  v = Vector{eltype(a)}(undef, length(a.lasts))
  v[1] = a.first
  v[2:end] .= @views(a.lasts[oneto(end - 1)]) .+ 1
  return v
end
@inline BlockArrays.blocklasts(a::BlockedUnitRange) = a.lasts

_diff(a::AbstractVector) = diff(a)
_diff(a::Tuple) = diff(collect(a))
@inline function BlockArrays.blocklengths(a::BlockedUnitRange)
  return if isempty(a.lasts)
    [_diff(a.lasts);]
  else
    [first(a.lasts) - a.first + 1; _diff(a.lasts)]
  end
end

function Base.length(a::BlockedUnitRange)
  return isempty(a.lasts) ? 0 : Integer(last(a.lasts) - a.first + 1)
end

Base.convert(::Type{BlockedUnitRange}, axis::BlockedUnitRange) = axis
function Base.convert(::Type{BlockedUnitRange}, axis::AbstractUnitRange{Int})
  return _BlockedUnitRange(first(axis), [last(axis)])
end
function Base.convert(::Type{BlockedUnitRange}, axis::Base.Slice)
  return _BlockedUnitRange(first(axis), [last(axis)])
end
function Base.convert(::Type{BlockedUnitRange}, axis::Base.IdentityUnitRange)
  return _BlockedUnitRange(first(axis), [last(axis)])
end
# TODO: Update.
Base.convert(::Type{BlockedUnitRange{CS}}, axis::BlockedUnitRange{CS}) where {CS} = axis
# TODO: Update.
function Base.convert(::Type{BlockedUnitRange{CS}}, axis::BlockedUnitRange) where {CS}
  return _BlockedUnitRange(first(axis), convert(CS, blocklasts(axis)))
end
# TODO: Update.
function Base.convert(::Type{BlockedUnitRange{CS}}, axis::AbstractUnitRange{Int}) where {CS}
  return convert(BlockedUnitRange{CS}, convert(BlockedUnitRange, axis))
end

Base.unitrange(b::BlockedUnitRange) = first(b):last(b)

# TODO: Update.
function Base.promote_rule(::Type{BlockedUnitRange{CS}}, ::Type{Base.OneTo{Int}}) where {CS}
  return UnitRange{Int}
end

BlockArrays.blockaxes(b::BlockedUnitRange) = _blockaxes(b.lasts)
_blockaxes(b::AbstractVector) = (Block.(axes(b, 1)),)
_blockaxes(b::Tuple) = (Block.(Base.OneTo(length(b))),)

@inline function BlockArrays.blockaxes(A::AbstractArray{T,N}, d) where {T,N}
  return d::Integer <= N ? blockaxes(A)[d] : Base.OneTo(1)
end

BlockArrays.blocksize(A) = map(length, blockaxes(A))
BlockArrays.blocksize(A, i) = length(blockaxes(A, i))
@inline BlockArrays.blocklength(t) = prod(blocksize(t))

BlockArrays.blocksizes(A) = map(blocklengths, axes(A))
BlockArrays.blocksizes(A, i) = blocklengths(axes(A, i))

Base.axes(b::BlockedUnitRange) = (_BlockedUnitRange(blocklasts(b) .- (first(b) - 1)),)
Base.unsafe_indices(b::BlockedUnitRange) = axes(b)
Base.first(b::BlockedUnitRange) = b.first
Base.last(b::BlockedUnitRange) = isempty(blocklasts(b)) ? first(b) - 1 : last(blocklasts(b))

# view and indexing are identical for a unitrange
Base.view(b::BlockedUnitRange, K::Block{1}) = b[K]

@propagate_inbounds function Base.getindex(b::BlockedUnitRange, K::Block{1})
  k = Integer(K)
  bax = blockaxes(b, 1)
  cs = blocklasts(b)
  @boundscheck K in bax || throw(BlockBoundsError(b, k))
  S = first(bax)
  K == S && return first(b):first(cs)
  return (cs[k - 1] + 1):cs[k]
end

@propagate_inbounds function Base.getindex(b::BlockedUnitRange, KR::BlockRange{1})
  cs = blocklasts(b)
  isempty(KR) && return _BlockedUnitRange(1, cs[1:0])
  K, J = first(KR), last(KR)
  k, j = Integer(K), Integer(J)
  bax = blockaxes(b, 1)
  @boundscheck K in bax || throw(BlockBoundsError(b, K))
  @boundscheck J in bax || throw(BlockBoundsError(b, J))
  K == first(bax) && return _BlockedUnitRange(first(b), cs[k:j])
  return _BlockedUnitRange(cs[k - 1] + 1, cs[k:j])
end

@propagate_inbounds function Base.getindex(
  b::BlockedUnitRange, KR::BlockRange{1,Tuple{Base.OneTo{Int}}}
)
  cs = blocklasts(b)
  isempty(KR) && return _BlockedUnitRange(1, cs[Base.OneTo(0)])
  J = last(KR)
  j = Integer(J)
  bax = blockaxes(b, 1)
  @boundscheck J in bax || throw(BlockBoundsError(b, J))
  return _BlockedUnitRange(first(b), cs[Base.OneTo(j)])
end

@propagate_inbounds Base.getindex(b::BlockedUnitRange, KR::BlockSlice) = b[KR.block]

_searchsortedfirst(a::AbstractVector, k) = searchsortedfirst(a, k)
function _searchsortedfirst(a::Tuple, k)
  k ≤ first(a) && return 1
  return 1 + _searchsortedfirst(tail(a), k)
end
_searchsortedfirst(a::Tuple{}, k) = 1

function BlockArrays.findblock(b::BlockedUnitRange, k::Integer)
  @boundscheck k in b || throw(BoundsError(b, k))
  return Block(_searchsortedfirst(blocklasts(b), k))
end

Base.dataids(b::BlockedUnitRange) = Base.dataids(blocklasts(b))

function Base.checkindex(::Type{Bool}, axis::BlockedUnitRange, ind::BlockIndexRange{1})
  return checkindex(Bool, axis, first(ind)) && checkindex(Bool, axis, last(ind))
end
function Base.checkindex(::Type{Bool}, axis::BlockedUnitRange, ind::BlockIndex{1})
  return checkindex(Bool, axis, block(ind)) &&
         checkbounds(Bool, axis[block(ind)], blockindex(ind))
end

Base.summary(a::BlockedUnitRange) = _block_summary(a)
Base.summary(io::IO, a::BlockedUnitRange) = _block_summary(io, a)

Base.axes(S::Base.Slice{<:BlockedUnitRange}) = (S.indices,)
Base.unsafe_indices(S::Base.Slice{<:BlockedUnitRange}) = (S.indices,)
Base.axes1(S::Base.Slice{<:BlockedUnitRange}) = S.indices

# This supports broadcasting with infinite block arrays
Base.BroadcastStyle(::Type{BlockedUnitRange{R}}) where {R} = Base.BroadcastStyle(R)

function BlockArrays.blockfirsts(a::BlockedUnitRange{Base.OneTo{Int}})
  a.first == 1 || error("Offset axes not supported")
  return Base.OneTo{Int}(length(a.lasts))
end
function BlockArrays.blocklengths(a::BlockedUnitRange{Base.OneTo{Int}})
  a.first == 1 || error("Offset axes not supported")
  return Ones{Int}(length(a.lasts))
end
function BlockArrays.blockfirsts(a::BlockedUnitRange{<:AbstractRange})
  st = step(a.lasts)
  a.first == 1 || error("Offset axes not supported")
  @assert first(a.lasts) - a.first + 1 == st
  return range(1; step=st, length=length(a.lasts))
end
function BlockArrys.blocklengths(a::BlockedUnitRange{<:AbstractRange})
  st = step(a.lasts)
  a.first == 1 || error("Offset axes not supported")
  @assert first(a.lasts) - a.first + 1 == st
  return Fill(st, length(a.lasts))
end
end
