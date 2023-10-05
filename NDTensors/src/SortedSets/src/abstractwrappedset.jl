# AbstractWrappedIndices: a wrapper around an `AbstractIndices`
# with methods automatically forwarded via `parent`
# and rewrapped via `rewrap`.
abstract type AbstractWrappedIndices{T,D} <: AbstractIndices{T} end

# Required interface
Base.parent(inds::AbstractWrappedIndices) = error("Not implemented")
function Dictionaries.empty_type(::Type{AbstractWrappedIndices{I}}, ::Type{I}) where {I}
  return error("Not implemented")
end
SmallVectors.thaw(::AbstractWrappedIndices) = error("Not implemented")
SmallVectors.freeze(::AbstractWrappedIndices) = error("Not implemented")
rewrap(::AbstractWrappedIndices, data) = error("Not implemented")

# Traits
SmallVectors.InsertStyle(::Type{<:AbstractWrappedIndices{T,D}}) where {T,D} = InsertStyle(D)

# AbstractIndices interface
@propagate_inbounds function Base.iterate(inds::AbstractWrappedIndices, state...)
  return iterate(parent(inds), state...)
end

# `I` is needed to avoid ambiguity error.
@inline Base.in(tag::I, inds::AbstractWrappedIndices{I}) where {I} = in(tag, parent(inds))
@inline Base.IteratorSize(inds::AbstractWrappedIndices) = Base.IteratorSize(parent(inds))
@inline Base.length(inds::AbstractWrappedIndices) = length(parent(inds))

@inline Dictionaries.istokenizable(inds::AbstractWrappedIndices) =
  istokenizable(parent(inds))
@inline Dictionaries.tokentype(inds::AbstractWrappedIndices) = tokentype(parent(inds))
@inline Dictionaries.iteratetoken(inds::AbstractWrappedIndices, s...) =
  iterate(parent(inds), s...)
@inline function Dictionaries.iteratetoken_reverse(inds::AbstractWrappedIndices)
  return iteratetoken_reverse(parent(inds))
end
@inline function Dictionaries.iteratetoken_reverse(inds::AbstractWrappedIndices, t)
  return iteratetoken_reverse(parent(inds), t)
end

@inline function Dictionaries.gettoken(inds::AbstractWrappedIndices, i)
  return gettoken(parent(inds), i)
end
@propagate_inbounds Dictionaries.gettokenvalue(inds::AbstractWrappedIndices, x) =
  gettokenvalue(parent(inds), x)

@inline Dictionaries.isinsertable(inds::AbstractWrappedIndices) = isinsertable(parent(inds))

# Specify `I` to fix ambiguity error.
@inline function Dictionaries.gettoken!(
  inds::AbstractWrappedIndices{I}, i::I, values=()
) where {I}
  return gettoken!(parent(inds), i, values)
end

@inline function Dictionaries.deletetoken!(inds::AbstractWrappedIndices, x, values=())
  deletetoken!(parent(inds), x, values)
  return inds
end

@inline function Base.empty!(inds::AbstractWrappedIndices, values=())
  empty!(parent(inds))
  return inds
end

# Not defined to be part of the `AbstractIndices` interface,
# but seems to be needed.
@inline function Base.filter!(pred, inds::AbstractWrappedIndices)
  filter!(pred, parent(inds))
  return inds
end

# TODO: Maybe require an implementation?
@inline function Base.copy(inds::AbstractWrappedIndices, eltype::Type)
  return typeof(inds)(copy(parent(inds), eltype))
end

# Not required for AbstractIndices interface but
# helps with faster code paths
SmallVectors.insert(inds::AbstractWrappedIndices, tag) = insert(parent(inds), tag)
Base.insert!(inds::AbstractWrappedIndices, tag) = insert!(parent(inds), tag)

SmallVectors.delete(inds::AbstractWrappedIndices, tag) = delete(parent(inds), tag)
Base.delete!(inds::AbstractWrappedIndices, tag) = delete!(parent(inds), tag)

function Base.union(inds1::AbstractWrappedIndices, inds2::AbstractWrappedIndices)
  return rewrap(inds1, union(parent(inds1), parent(inds2)))
end
function Base.union(inds1::AbstractWrappedIndices, inds2)
  return rewrap(inds1, union(parent(inds1), inds2))
end

function Base.intersect(inds1::AbstractWrappedIndices, inds2::AbstractWrappedIndices)
  return rewrap(inds1, intersect(parent(inds1), parent(inds2)))
end
function Base.intersect(inds1::AbstractWrappedIndices, inds2)
  return rewrap(inds1, intersect(parent(inds1), inds2))
end

function Base.setdiff(inds1::AbstractWrappedIndices, inds2::AbstractWrappedIndices)
  return rewrap(inds1, setdiff(parent(inds1), parent(inds2)))
end
function Base.setdiff(inds1::AbstractWrappedIndices, inds2)
  return rewrap(inds1, setdiff(parent(inds1), inds2))
end

function Base.symdiff(inds1::AbstractWrappedIndices, inds2::AbstractWrappedIndices)
  return rewrap(inds1, symdiff(parent(inds1), parent(inds2)))
end
function Base.symdiff(inds1::AbstractWrappedIndices, inds2)
  return rewrap(inds1, symdiff(parent(inds1), inds2))
end
