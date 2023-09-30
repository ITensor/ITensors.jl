"""
    SortedIndices(iter)

Construct an `SortedIndices <: AbstractIndices` from an arbitrary Julia iterable with unique
elements. Lookup uses that they are sorted.

SortedIndices can be faster than ArrayIndices which use naive search that may be optimal for
small collections. Larger collections are better handled by containers like `Indices`.
"""
struct SortedIndices{I,Inds<:AbstractArray{I},SortKwargs<:NamedTuple} <: AbstractSet{I}
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

# Traits
@inline SmallVectors.InsertStyle(::Type{<:SortedIndices{I,Inds}}) where {I,Inds} =
  InsertStyle(Inds)
@inline SmallVectors.thaw(i::SortedIndices) =
  SortedIndices(thaw(i.inds); checksorted=false, checkunique=false, i.sort_kwargs...)
@inline SmallVectors.freeze(i::SortedIndices) =
  SortedIndices(freeze(i.inds); checksorted=false, checkunique=false, i.sort_kwargs...)

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
  return insorted(i, parent(inds); inds.sort_kwargs...)
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
  r = searchsorted(a, i; inds.sort_kwargs...)
  @assert 0 ≤ length(r) ≤ 1 # If > 1, means the elements are not unique
  length(r) == 0 && return (false, 0)
  return (true, convert(Int, only(r)))
end
@propagate_inbounds Dictionaries.gettokenvalue(inds::SortedIndices, x::Int) =
  parent(inds)[x]

@inline Dictionaries.isinsertable(i::SortedIndices) = isinsertable(parent(inds))

@inline function Dictionaries.gettoken!(inds::SortedIndices{I}, i::I, values=()) where {I}
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
@inline Dictionaries.empty_type(::Type{SortedIndices{I,D}}, ::Type{I}) where {I,D} =
  SortedIndices{I,empty_type(D, I)}

@inline function Base.copy(inds::SortedIndices, ::Type{I}) where {I}
  if I === eltype(inds)
    SortedIndices{I}(copy(parent(inds)); checkunique=false, checksorted=false)
  else
    SortedIndices{I}(
      convert(AbstractArray{I}, parent(inds)); checkunique=false, checksorted=false
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
  return inds
end

# Custom faster operations (not required for interface)
@inline function Base.union!(vec::SortedIndices, items)
  for item in items
    insert!(vec, item)
  end
  return vec
end

function Base.union(vec::SortedIndices, items)
  return union(InsertStyle(vec), vec, items)
  error("Not implemented")
  r = searchsorted(vec, item; kwargs...)
  if length(r) == 0
    vec = insert(vec, first(r), item)
  end
  return vec
end

# TODO: Use `insertsortedunique`, `mergesortedunique`
# from `SmallVectors`.
function Base.union(::FastCopy, i::SortedIndices, itr)
  inds = SmallVectors.mergesortedunique(parent(i), itr; i.sort_kwargs...)
  return SortedIndices(inds; checksorted=false, checkunique=false, i.sort_kwargs...)
end
