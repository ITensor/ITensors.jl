"""
    SortedIndices(iter)

Construct an `SortedIndices <: AbstractIndices` from an arbitrary Julia iterable with unique
elements. Lookup uses that they are sorted.

SortedIndices can be faster than ArrayIndices which use naive search that may be optimal for
small collections. Larger collections are better handled by containers like `Indices`.
"""
struct SortedIndices{I,Inds<:AbstractArray{I},Order<:Ordering} <: AbstractSet{I}
  inds::Inds
  order::Order
  global @inline _SortedIndices(
    inds::Inds, order::Order
  ) where {I,Inds<:AbstractArray{I},Order<:Ordering} = new{I,Inds,Order}(inds, order)
end

# Inner constructor
function SortedIndices{I,Inds,Order}(
  a::Inds, order::Order; issorted=issorted, allunique=allunique
) where {I,Inds<:AbstractArray{I},Order<:Ordering}
  if !issorted(a, order)
    a = sort(a, order)
  end
  if !alluniquesorted(a, order)
    a = uniquesorted(a, order)
  end
  return _SortedIndices(a, order)
end

@inline function SortedIndices{I,Inds,Order}(
  a::AbstractArray, order::Ordering; issorted=issorted, allunique=allunique
) where {I,Inds<:AbstractArray{I},Order<:Ordering}
  return SortedIndices{I,Inds,Order}(
    convert(Inds, a), convert(Order, order); issorted, allunique
  )
end

@inline function SortedIndices{I,Inds}(
  a::AbstractArray, order::Order; issorted=issorted, allunique=allunique
) where {I,Inds<:AbstractArray{I},Order<:Ordering}
  return SortedIndices{I,Inds,Order}(a, order; issorted, allunique)
end

@inline function SortedIndices(
  a::Inds, order::Ordering; issorted=issorted, allunique=allunique
) where {I,Inds<:AbstractArray{I}}
  return SortedIndices{I,Inds}(a, order; issorted, allunique)
end

@inline function SortedIndices{I,Inds}(
  a::Inds;
  lt=isless,
  by=identity,
  rev::Bool=false,
  order::Ordering=Forward,
  issorted=issorted,
  allunique=allunique,
) where {I,Inds<:AbstractArray{I}}
  order = ord(lt, by, rev, order)
  return SortedIndices{I,Inds}(a, order; issorted, allunique)
end

const SortedSet = SortedIndices

# Traits
@inline SmallVectors.InsertStyle(::Type{<:SortedIndices{I,Inds}}) where {I,Inds} =
  InsertStyle(Inds)
@inline SmallVectors.thaw(i::SortedIndices) = SortedIndices(thaw(i.inds), i.order)
@inline SmallVectors.freeze(i::SortedIndices) = SortedIndices(freeze(i.inds), i.order)

@propagate_inbounds SortedIndices(; kwargs...) = SortedIndices{Any}([]; kwargs...)
@propagate_inbounds SortedIndices{I}(; kwargs...) where {I} =
  SortedIndices{I,Vector{I}}(I[]; kwargs...)
@propagate_inbounds SortedIndices{I,Inds}(; kwargs...) where {I,Inds} =
  SortedIndices{I}(Inds(); kwargs...)

@propagate_inbounds SortedIndices(iter; kwargs...) = SortedIndices(collect(iter); kwargs...)
@propagate_inbounds SortedIndices{I}(iter; kwargs...) where {I} =
  SortedIndices{I}(collect(I, iter); kwargs...)

@propagate_inbounds SortedIndices(a::AbstractArray{I}; kwargs...) where {I} =
  SortedIndices{I}(a; kwargs...)
@propagate_inbounds SortedIndices{I}(a::AbstractArray{I}; kwargs...) where {I} =
  SortedIndices{I,typeof(a)}(a; kwargs...)

@propagate_inbounds SortedIndices{I,Inds}(
  a::AbstractArray; kwargs...
) where {I,Inds<:AbstractArray{I}} = SortedIndices{I,Inds}(Inds(a); kwargs...)

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

@inline Base.parent(inds::SortedIndices) = getfield(inds, :inds)

# Basic interface
@propagate_inbounds function Base.iterate(i::SortedIndices{I}, state...) where {I}
  return iterate(parent(i), state...)
end

@inline function Base.in(i::I, inds::SortedIndices{I}) where {I}
  return _insorted(i, parent(inds), inds.order)
end
@inline Base.IteratorSize(::SortedIndices) = Base.HasLength()
@inline Base.length(inds::SortedIndices) = length(parent(inds))

@inline Dictionaries.istokenizable(i::SortedIndices) = true
@inline Dictionaries.tokentype(::SortedIndices) = Int
@inline Dictionaries.iteratetoken(inds::SortedIndices, s...) =
  iterate(LinearIndices(parent(inds)), s...)
@inline function Dictionaries.iteratetoken_reverse(inds::SortedIndices)
  li = LinearIndices(parent(inds))
  if isempty(li)
    return nothing
  else
    t = last(li)
    return (t, t)
  end
end
@inline function Dictionaries.iteratetoken_reverse(inds::SortedIndices, t)
  li = LinearIndices(parent(inds))
  t -= 1
  if t < first(li)
    return nothing
  else
    return (t, t)
  end
end

@inline function Dictionaries.gettoken(inds::SortedIndices, i)
  a = parent(inds)
  r = searchsorted(a, i, inds.order)
  @assert 0 ≤ length(r) ≤ 1 # If > 1, means the elements are not unique
  length(r) == 0 && return (false, 0)
  return (true, convert(Int, only(r)))
end
@propagate_inbounds Dictionaries.gettokenvalue(inds::SortedIndices, x::Int) =
  parent(inds)[x]

@inline Dictionaries.isinsertable(i::SortedIndices) = isinsertable(parent(inds))

@inline function Dictionaries.gettoken!(inds::SortedIndices{I}, i::I, values=()) where {I}
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

@inline function Dictionaries.deletetoken!(inds::SortedIndices, x::Int, values=())
  deleteat!(parent(inds), x)
  foreach(v -> deleteat!(v, x), values)
  return inds
end

@inline function Base.empty!(inds::SortedIndices, values=())
  empty!(parent(inds))
  foreach(empty!, values)
  return inds
end

# TODO: Make into `MSmallVector`?
# More generally, make a `thaw(::AbstractArray)` function to return
# a mutable version of an AbstractArray.
@inline Dictionaries.empty_type(
  ::Type{SortedIndices{I,D,Order}}, ::Type{I}
) where {I,D,Order} = SortedIndices{I,Dictionaries.empty_type(D, I),Order}

@inline Dictionaries.empty_type(::Type{<:AbstractVector}, ::Type{I}) where {I} = Vector{I}

function Base.empty(inds::SortedIndices{I,D}, ::Type{I}) where {I,D}
  return Dictionaries.empty_type(typeof(inds), I)(D(), inds.order)
end

@inline function Base.copy(inds::SortedIndices, ::Type{I}) where {I}
  if I === eltype(inds)
    SortedIndices(
      copy(parent(inds)), inds.order; issorted=Returns(true), allunique=Returns(true)
    )
  else
    SortedIndices(
      convert(AbstractArray{I}, parent(inds)),
      inds.order;
      issorted=Returns(true),
      allunique=Returns(true),
    )
  end
end

# TODO: Can this take advantage of sorting?
@inline function Base.filter!(pred, inds::SortedIndices)
  filter!(pred, parent(inds))
  return inds
end

function Dictionaries.randtoken(rng::Random.AbstractRNG, inds::SortedIndices)
  return rand(rng, keys(parent(inds)))
end

@inline function Base.sort!(
  inds::SortedIndices; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward
)
  # No-op, should be sorted already.
  # TODO: Check `ord(lt, by, rev, order) == inds.ord`.
  return inds
end

# Custom faster operations (not required for interface)
function Base.union!(inds::SortedIndices, items::SortedIndices)
  if inds.order ≠ items.order
    # Reorder if the orderings are different.
    items = SortedIndices(parent(inds), inds.order)
  end
  unionsortedunique!(parent(inds), parent(items), inds.order)
  return inds
end

function Base.union(inds::SortedIndices, items::SortedIndices)
  if inds.order ≠ items.order
    # Reorder if the orderings are different.
    items = SortedIndices(parent(inds), inds.order)
  end
  out = unionsortedunique(parent(inds), parent(items), inds.order)
  return SortedIndices(out, inds.order; issorted=Returns(true), allunique=Returns(true))
end

function Base.union(inds::SortedIndices, items)
  return union(inds, SortedIndices(items, inds.order))
end

function Base.intersect(inds::SortedIndices, items::SortedIndices)
  # TODO: Make an `intersectsortedunique`.
  return intersect(NotInsertable(), inds, items)
end

function Base.setdiff(inds::SortedIndices, items)
  return setdiff(inds, SortedIndices(items, inds.order))
end

function Base.setdiff(inds::SortedIndices, items::SortedIndices)
  # TODO: Make an `setdiffsortedunique`.
  return setdiff(NotInsertable(), inds, items)
end

function Base.symdiff(inds::SortedIndices, items)
  return symdiff(inds, SortedIndices(items, inds.order))
end

function Base.symdiff(inds::SortedIndices, items::SortedIndices)
  # TODO: Make an `symdiffsortedunique`.
  return symdiff(NotInsertable(), inds, items)
end
