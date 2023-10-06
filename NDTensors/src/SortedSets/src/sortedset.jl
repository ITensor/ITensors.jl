"""
    SortedSet(iter)

Construct an `SortedSet <: AbstractIndices` from an arbitrary Julia iterable with unique
elements. Lookup uses that they are sorted.

SortedSet can be faster than ArrayIndices which use naive search that may be optimal for
small collections. Larger collections are better handled by containers like `Indices`.
"""
struct SortedSet{I,Inds<:AbstractArray{I},Order<:Ordering} <: AbstractSet{I}
  inds::Inds
  order::Order
  global @inline _SortedSet(
    inds::Inds, order::Order
  ) where {I,Inds<:AbstractArray{I},Order<:Ordering} = new{I,Inds,Order}(inds, order)
end

# Dictionaries.jl interface
const SortedIndices = SortedSet

# Inner constructor.
# Assumes it is already sorted.
function SortedSet{I,Inds,Order}(
  a::Inds, order::Order
) where {I,Inds<:AbstractArray{I},Order<:Ordering}
  return _SortedSet(a, order)
end

@inline function SortedSet{I,Inds,Order}(
  a::AbstractArray, order::Ordering
) where {I,Inds<:AbstractArray{I},Order<:Ordering}
  return SortedSet{I,Inds,Order}(convert(Inds, a), convert(Order, order))
end

@inline function SortedSet{I,Inds}(
  a::AbstractArray, order::Order
) where {I,Inds<:AbstractArray{I},Order<:Ordering}
  return SortedSet{I,Inds,Order}(a, order)
end

@inline function SortedSet(a::Inds, order::Ordering) where {I,Inds<:AbstractArray{I}}
  return SortedSet{I,Inds}(a, order)
end

@inline function SortedSet{I,Inds}(
  a::Inds; lt=isless, by=identity, rev::Bool=false, issorted=issorted, allunique=allunique
) where {I,Inds<:AbstractArray{I}}
  if !issorted(a; lt, by, rev)
    a = sort(a; lt, by, rev)
  end
  if !alluniquesorted(a; lt, by, rev)
    a = uniquesorted(a, order)
  end
  new_order = ord(lt, by, rev)
  return SortedSet{I,Inds}(a, new_order)
end

# Traits
@inline SmallVectors.InsertStyle(::Type{<:SortedSet{I,Inds}}) where {I,Inds} =
  InsertStyle(Inds)
@inline SmallVectors.thaw(i::SortedSet) = SortedSet(thaw(i.inds), i.order)
@inline SmallVectors.freeze(i::SortedSet) = SortedSet(freeze(i.inds), i.order)

@propagate_inbounds SortedSet(; kwargs...) = SortedSet{Any}([]; kwargs...)
@propagate_inbounds SortedSet{I}(; kwargs...) where {I} =
  SortedSet{I,Vector{I}}(I[]; kwargs...)
@propagate_inbounds SortedSet{I,Inds}(; kwargs...) where {I,Inds} =
  SortedSet{I}(Inds(); kwargs...)

@propagate_inbounds SortedSet(iter; kwargs...) = SortedSet(collect(iter); kwargs...)
@propagate_inbounds SortedSet{I}(iter; kwargs...) where {I} =
  SortedSet{I}(collect(I, iter); kwargs...)

@propagate_inbounds SortedSet(a::AbstractArray{I}; kwargs...) where {I} =
  SortedSet{I}(a; kwargs...)
@propagate_inbounds SortedSet{I}(a::AbstractArray{I}; kwargs...) where {I} =
  SortedSet{I,typeof(a)}(a; kwargs...)

@propagate_inbounds SortedSet{I,Inds}(
  a::AbstractArray; kwargs...
) where {I,Inds<:AbstractArray{I}} = SortedSet{I,Inds}(Inds(a); kwargs...)

function Base.convert(::Type{AbstractIndices{I}}, inds::SortedSet) where {I}
  return convert(SortedSet{I}, inds)
end
function Base.convert(::Type{SortedSet}, inds::AbstractIndices{I}) where {I}
  return convert(SortedSet{I}, inds)
end
function Base.convert(::Type{SortedSet{I}}, inds::AbstractIndices) where {I}
  return convert(SortedSet{I,Vector{I}}, inds)
end
function Base.convert(
  ::Type{SortedSet{I,Inds}}, inds::AbstractIndices
) where {I,Inds<:AbstractArray{I}}
  a = convert(Inds, collect(I, inds))
  return @inbounds SortedSet{I,typeof(a)}(a)
end

Base.convert(::Type{SortedSet{I}}, inds::SortedSet{I}) where {I} = inds
function Base.convert(
  ::Type{SortedSet{I}}, inds::SortedSet{<:Any,Inds}
) where {I,Inds<:AbstractArray{I}}
  return convert(SortedSet{I,Inds}, inds)
end
function Base.convert(
  ::Type{SortedSet{I,Inds}}, inds::SortedSet{I,Inds}
) where {I,Inds<:AbstractArray{I}}
  return inds
end
function Base.convert(
  ::Type{SortedSet{I,Inds}}, inds::SortedSet
) where {I,Inds<:AbstractArray{I}}
  a = convert(Inds, parent(inds))
  return @inbounds SortedSet{I,Inds}(a)
end

@inline Base.parent(inds::SortedSet) = getfield(inds, :inds)

# Basic interface
@propagate_inbounds function Base.iterate(i::SortedSet{I}, state...) where {I}
  return iterate(parent(i), state...)
end

@inline function Base.in(i::I, inds::SortedSet{I}) where {I}
  return _insorted(i, parent(inds), inds.order)
end
@inline Base.IteratorSize(::SortedSet) = Base.HasLength()
@inline Base.length(inds::SortedSet) = length(parent(inds))

@inline Dictionaries.istokenizable(i::SortedSet) = true
@inline Dictionaries.tokentype(::SortedSet) = Int
@inline Dictionaries.iteratetoken(inds::SortedSet, s...) =
  iterate(LinearIndices(parent(inds)), s...)
@inline function Dictionaries.iteratetoken_reverse(inds::SortedSet)
  li = LinearIndices(parent(inds))
  if isempty(li)
    return nothing
  else
    t = last(li)
    return (t, t)
  end
end
@inline function Dictionaries.iteratetoken_reverse(inds::SortedSet, t)
  li = LinearIndices(parent(inds))
  t -= 1
  if t < first(li)
    return nothing
  else
    return (t, t)
  end
end

@inline function Dictionaries.gettoken(inds::SortedSet, i)
  a = parent(inds)
  r = searchsorted(a, i, inds.order)
  @assert 0 ≤ length(r) ≤ 1 # If > 1, means the elements are not unique
  length(r) == 0 && return (false, 0)
  return (true, convert(Int, only(r)))
end
@propagate_inbounds Dictionaries.gettokenvalue(inds::SortedSet, x::Int) = parent(inds)[x]

@inline Dictionaries.isinsertable(i::SortedSet) = isinsertable(parent(inds))

@inline function Dictionaries.gettoken!(inds::SortedSet{I}, i::I, values=()) where {I}
  a = parent(inds)
  r = searchsorted(a, i, inds.order)
  @assert 0 ≤ length(r) ≤ 1 # If > 1, means the elements are not unique
  if length(r) == 0
    insert!(a, first(r), i)
    foreach(v -> resize!(v, length(v) + 1), values)
    return (false, last(LinearIndices(a)))
  end
  return (true, convert(Int, only(r)))
end

@inline function Dictionaries.deletetoken!(inds::SortedSet, x::Int, values=())
  deleteat!(parent(inds), x)
  foreach(v -> deleteat!(v, x), values)
  return inds
end

@inline function Base.empty!(inds::SortedSet, values=())
  empty!(parent(inds))
  foreach(empty!, values)
  return inds
end

# TODO: Make into `MSmallVector`?
# More generally, make a `thaw(::AbstractArray)` function to return
# a mutable version of an AbstractArray.
@inline Dictionaries.empty_type(::Type{SortedSet{I,D,Order}}, ::Type{I}) where {I,D,Order} =
  SortedSet{I,Dictionaries.empty_type(D, I),Order}

@inline Dictionaries.empty_type(::Type{<:AbstractVector}, ::Type{I}) where {I} = Vector{I}

function Base.empty(inds::SortedSet{I,D}, ::Type{I}) where {I,D}
  return Dictionaries.empty_type(typeof(inds), I)(D(), inds.order)
end

@inline function Base.copy(inds::SortedSet, ::Type{I}) where {I}
  if I === eltype(inds)
    SortedSet(copy(parent(inds)), inds.order)
  else
    SortedSet(convert(AbstractArray{I}, parent(inds)), inds.order)
  end
end

# TODO: Can this take advantage of sorting?
@inline function Base.filter!(pred, inds::SortedSet)
  filter!(pred, parent(inds))
  return inds
end

function Dictionaries.randtoken(rng::Random.AbstractRNG, inds::SortedSet)
  return rand(rng, keys(parent(inds)))
end

@inline function Base.sort!(inds::SortedSet; lt=isless, by=identity, rev::Bool=false)
  # No-op, should be sorted already.
  # TODO: Check `ord(lt, by, rev, order) == inds.ord`.
  return inds
end

# Custom faster operations (not required for interface)
function Base.union!(inds::SortedSet, items::SortedSet)
  if inds.order ≠ items.order
    # Reorder if the orderings are different.
    items = SortedSet(parent(inds), inds.order)
  end
  unionsortedunique!(parent(inds), parent(items), inds.order)
  return inds
end

function Base.union(inds::SortedSet, items::SortedSet)
  if inds.order ≠ items.order
    # Reorder if the orderings are different.
    items = SortedSet(parent(inds), inds.order)
  end
  out = unionsortedunique(parent(inds), parent(items), inds.order)
  return SortedSet(out, inds.order)
end

function Base.union(inds::SortedSet, items)
  return union(inds, SortedSet(items, inds.order))
end

function Base.intersect(inds::SortedSet, items::SortedSet)
  # TODO: Make an `intersectsortedunique`.
  return intersect(NotInsertable(), inds, items)
end

function Base.setdiff(inds::SortedSet, items)
  return setdiff(inds, SortedSet(items, inds.order))
end

function Base.setdiff(inds::SortedSet, items::SortedSet)
  # TODO: Make an `setdiffsortedunique`.
  return setdiff(NotInsertable(), inds, items)
end

function Base.symdiff(inds::SortedSet, items)
  return symdiff(inds, SortedSet(items, inds.order))
end

function Base.symdiff(inds::SortedSet, items::SortedSet)
  # TODO: Make an `symdiffsortedunique`.
  return symdiff(NotInsertable(), inds, items)
end
