# AbstractWrappedSet: a wrapper around an `AbstractIndices`
# with methods automatically forwarded via `parent`
# and rewrapped via `rewrap`.
abstract type AbstractWrappedSet{T,D} <: AbstractIndices{T} end

# Required interface
Base.parent(set::AbstractWrappedSet) = error("Not implemented")
function Dictionaries.empty_type(::Type{AbstractWrappedSet{I}}, ::Type{I}) where {I}
  return error("Not implemented")
end
rewrap(::AbstractWrappedSet, data) = error("Not implemented")

SmallVectors.thaw(set::AbstractWrappedSet) = rewrap(set, thaw(parent(set)))
SmallVectors.freeze(set::AbstractWrappedSet) = rewrap(set, freeze(parent(set)))

# Traits
SmallVectors.InsertStyle(::Type{<:AbstractWrappedSet{T,D}}) where {T,D} = InsertStyle(D)

# AbstractSet interface
@propagate_inbounds function Base.iterate(set::AbstractWrappedSet, state...)
  return iterate(parent(set), state...)
end

# `I` is needed to avoid ambiguity error.
@inline Base.in(item::I, set::AbstractWrappedSet{I}) where {I} = in(item, parent(set))
@inline Base.IteratorSize(set::AbstractWrappedSet) = Base.IteratorSize(parent(set))
@inline Base.length(set::AbstractWrappedSet) = length(parent(set))

@inline Dictionaries.istokenizable(set::AbstractWrappedSet) = istokenizable(parent(set))
@inline Dictionaries.tokentype(set::AbstractWrappedSet) = tokentype(parent(set))
@inline Dictionaries.iteratetoken(set::AbstractWrappedSet, s...) = iterate(
  parent(set), s...
)
@inline function Dictionaries.iteratetoken_reverse(set::AbstractWrappedSet)
  return iteratetoken_reverse(parent(set))
end
@inline function Dictionaries.iteratetoken_reverse(set::AbstractWrappedSet, t)
  return iteratetoken_reverse(parent(set), t)
end

@inline function Dictionaries.gettoken(set::AbstractWrappedSet, i)
  return gettoken(parent(set), i)
end
@propagate_inbounds Dictionaries.gettokenvalue(set::AbstractWrappedSet, x) = gettokenvalue(
  parent(set), x
)

@inline Dictionaries.isinsertable(set::AbstractWrappedSet) = isinsertable(parent(set))

# Specify `I` to fix ambiguity error.
@inline function Dictionaries.gettoken!(
  set::AbstractWrappedSet{I}, i::I, values=()
) where {I}
  return gettoken!(parent(set), i, values)
end

@inline function Dictionaries.deletetoken!(set::AbstractWrappedSet, x, values=())
  deletetoken!(parent(set), x, values)
  return set
end

@inline function Base.empty!(set::AbstractWrappedSet, values=())
  empty!(parent(set))
  return set
end

# Not defined to be part of the `AbstractIndices` interface,
# but seems to be needed.
@inline function Base.filter!(pred, set::AbstractWrappedSet)
  filter!(pred, parent(set))
  return set
end

# TODO: Maybe require an implementation?
@inline function Base.copy(set::AbstractWrappedSet, eltype::Type)
  return typeof(set)(copy(parent(set), eltype))
end

# Not required for AbstractIndices interface but
# helps with faster code paths
SmallVectors.insert(set::AbstractWrappedSet, item) = rewrap(set, insert(parent(set), item))
function Base.insert!(set::AbstractWrappedSet, item)
  insert!(parent(set), item)
  return set
end

SmallVectors.delete(set::AbstractWrappedSet, item) = rewrap(set, delete(parent(set), item))
function Base.delete!(set::AbstractWrappedSet, item)
  delete!(parent(set), item)
  return set
end

function Base.union(set1::AbstractWrappedSet, set2::AbstractWrappedSet)
  return rewrap(set1, union(parent(set1), parent(set2)))
end
function Base.union(set1::AbstractWrappedSet, set2)
  return rewrap(set1, union(parent(set1), set2))
end

function Base.intersect(set1::AbstractWrappedSet, set2::AbstractWrappedSet)
  return rewrap(set1, intersect(parent(set1), parent(set2)))
end
function Base.intersect(set1::AbstractWrappedSet, set2)
  return rewrap(set1, intersect(parent(set1), set2))
end

function Base.setdiff(set1::AbstractWrappedSet, set2::AbstractWrappedSet)
  return rewrap(set1, setdiff(parent(set1), parent(set2)))
end
function Base.setdiff(set1::AbstractWrappedSet, set2)
  return rewrap(set1, setdiff(parent(set1), set2))
end

function Base.symdiff(set1::AbstractWrappedSet, set2::AbstractWrappedSet)
  return rewrap(set1, symdiff(parent(set1), parent(set2)))
end
function Base.symdiff(set1::AbstractWrappedSet, set2)
  return rewrap(set1, symdiff(parent(set1), set2))
end
