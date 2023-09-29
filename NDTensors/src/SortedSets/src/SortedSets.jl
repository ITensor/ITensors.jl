module SortedSets
using Dictionaries

using Base: @propagate_inbounds
using Base.Order: Ordering, Forward
using Random

import Dictionaries:
  istokenizable,
  tokentype,
  iteratetoken,
  iteratetoken_reverse,
  gettoken,
  gettokenvalue,
  isinsertable,
  gettoken!,
  empty_type,
  deletetoken!,
  randtoken

export SortedSet

# TODO:
# Make an `AbstractSortedIndices`? Is that needed?
# Define specialized implementations for:
#
# Base.union
# Base.intersect
# Base.setdiff
# Base.symdiff
# Base.sort
#
# which can be dispatched on `SmallVector` for faster operations,
# potentially using a trait (`HasFastCopy`?).

"""
    SortedIndices(iter)

Construct an `SortedIndices <: AbstractIndices` from an arbitrary Julia iterable with unique
elements. Lookup uses that they are sorted.

SortedIndices can be faster than ArrayIndices which use naive search that may be optimal for
small collections. Larger collections are better handled by containers like `Indices`.
"""
struct SortedIndices{I,Inds<:AbstractArray{I},SortKwargs<:NamedTuple} <: AbstractIndices{I}
  inds::Inds
  sort_kwargs::SortKwargs
  @inline function SortedIndices{I,Inds}(
    a::Inds;
    lt=isless,
    by=identity,
    rev::Bool=false,
    order::Ordering=Forward,
    checksorted::Bool=true,
    checkunique::Bool=true,
  ) where {I,Inds<:AbstractArray{I}}
    if checkunique
      @assert allunique(Iterators.map(by, a))
    end
    if checksorted
      @assert issorted(a; lt, by, rev, order)
    end
    sort_kwargs = (; lt, by, rev, order)
    return new{I,Inds,typeof(sort_kwargs)}(a, sort_kwargs)
  end
end

const SortedSet = SortedIndices

@propagate_inbounds SortedIndices() = SortedIndices{Any}([])
@propagate_inbounds SortedIndices{I}() where {I} = SortedIndices{I,Vector{I}}(I[])
@propagate_inbounds SortedIndices{I,Inds}() where {I,Inds} = SortedIndices{I}(Inds())

@propagate_inbounds SortedIndices(iter) = SortedIndices(collect(iter))
@propagate_inbounds SortedIndices{I}(iter) where {I} = SortedIndices{I}(collect(I, iter))

@propagate_inbounds SortedIndices(a::AbstractArray{I}) where {I} = SortedIndices{I}(a)
@propagate_inbounds SortedIndices{I}(a::AbstractArray{I}) where {I} =
  SortedIndices{I,typeof(a)}(a)

function Base.convert(::Type{AbstractIndices{I}}, inds::SortedIndices) where {I}
  return convert(SortedIndices{I}, inds)
end
function Base.convert(::Type{SortedIndices}, inds::AbstractIndices{I}) where {I}
  return convert(SortedIndices{I}, inds)
end
function Base.convert(::Type{SortedIndices{I}}, inds::AbstractIndices) where {I}
  return convert(SortedIndices{I,Vector{I}}, inds)
end
function Base.convert(
  ::Type{SortedIndices{I,Inds}}, inds::AbstractIndices
) where {I,Inds<:AbstractArray{I}}
  a = convert(Inds, collect(I, inds))
  return @inbounds SortedIndices{I,typeof(a)}(a)
end

Base.convert(::Type{SortedIndices{I}}, inds::SortedIndices{I}) where {I} = inds
function Base.convert(
  ::Type{SortedIndices{I}}, inds::SortedIndices{<:Any,Inds}
) where {I,Inds<:AbstractArray{I}}
  return convert(SortedIndices{I,Inds}, inds)
end
function Base.convert(
  ::Type{SortedIndices{I,Inds}}, inds::SortedIndices{I,Inds}
) where {I,Inds<:AbstractArray{I}}
  return inds
end
function Base.convert(
  ::Type{SortedIndices{I,Inds}}, inds::SortedIndices
) where {I,Inds<:AbstractArray{I}}
  a = convert(Inds, parent(inds))
  return @inbounds SortedIndices{I,Inds}(a)
end

Base.parent(inds::SortedIndices) = getfield(inds, :inds)

# Basic interface
@propagate_inbounds function Base.iterate(i::SortedIndices{I}, state...) where {I}
  return iterate(parent(i), state...)
end

function Base.in(i::I, inds::SortedIndices{I}) where {I}
  return insorted(i, parent(inds); inds.sort_kwargs...)
end
Base.IteratorSize(::SortedIndices) = Base.HasLength()
Base.length(inds::SortedIndices) = length(parent(inds))

istokenizable(i::SortedIndices) = true
tokentype(::SortedIndices) = Int
@inline iteratetoken(inds::SortedIndices, s...) = iterate(LinearIndices(parent(inds)), s...)
@inline function iteratetoken_reverse(inds::SortedIndices)
  li = LinearIndices(parent(inds))
  if isempty(li)
    return nothing
  else
    t = last(li)
    return (t, t)
  end
end
@inline function iteratetoken_reverse(inds::SortedIndices, t)
  li = LinearIndices(parent(inds))
  t -= 1
  if t < first(li)
    return nothing
  else
    return (t, t)
  end
end

@inline function gettoken(inds::SortedIndices, i)
  a = parent(inds)
  r = searchsorted(a, i; inds.sort_kwargs...)
  @assert 0 ≤ length(r) ≤ 1 # If > 1, means the elements are not unique
  length(r) == 0 && return (false, 0)
  return (true, convert(Int, only(r)))
end
@propagate_inbounds gettokenvalue(inds::SortedIndices, x::Int) = parent(inds)[x]

isinsertable(i::SortedIndices) = true # Need an array trait here...

## # For `SmallVector`
## # TODO: Make this more general, based on a trait?
## isinsertable(i::SortedIndices{<:Any,<:SmallVector}) = false

@inline function gettoken!(inds::SortedIndices{I}, i::I, values=()) where {I}
  a = parent(inds)
  r = searchsorted(a, i; inds.sort_kwargs...)
  @assert 0 ≤ length(r) ≤ 1 # If > 1, means the elements are not unique
  if length(r) == 0
    insert!(a, first(r), i)
    foreach(v -> resize!(v, length(v) + 1), values)
    return (false, last(LinearIndices(a)))
  end
  return (true, convert(Int, only(r)))
end

@inline function deletetoken!(inds::SortedIndices, x::Int, values=())
  deleteat!(parent(inds), x)
  foreach(v -> deleteat!(v, x), values)
  return inds
end

function Base.empty!(inds::SortedIndices, values=())
  empty!(parent(inds))
  foreach(empty!, values)
  return inds
end

# TODO: Make into `MSmallVector`?
empty_type(::Type{<:SortedIndices}, ::Type{I}) where {I} = SortedIndices{I,Vector{I}}

function Base.copy(inds::SortedIndices, ::Type{I}) where {I}
  if I === eltype(inds)
    # TODO: Disable checking unique and sorted.
    SortedIndices{I}(copy(parent(inds)))
  else
    # TODO: Disable checking unique and sorted.
    SortedIndices{I}(convert(AbstractArray{I}, parent(inds)))
  end
end

# TODO: Can this take advantage of sorting?
function Base.filter!(pred, inds::SortedIndices)
  filter!(pred, parent(inds))
  return inds
end

function randtoken(rng::Random.AbstractRNG, inds::SortedIndices)
  return rand(rng, keys(parent(inds)))
end

function Base.sort!(inds::SortedIndices; kwargs...)
  # TODO: No-op, should be sorted already.
  sort!(inds.inds; kwargs...)
  return inds
end
end
