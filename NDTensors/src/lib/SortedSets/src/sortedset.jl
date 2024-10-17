"""
    SortedSet(iter)

Construct an `SortedSet <: AbstractSet` from an arbitrary Julia iterable with unique
elements. Lookup uses that they are sorted.

SortedSet can be faster than ArrayIndices which use naive search that may be optimal for
small collections. Larger collections are better handled by containers like `Indices`.
"""
struct SortedSet{T,Data<:AbstractArray{T},Order<:Ordering} <: AbstractSet{T}
  data::Data
  order::Order
  global @inline _SortedSet(data::Data, order::Order) where {T,Data<:AbstractArray{T},Order<:Ordering} = new{
    T,Data,Order
  }(
    data, order
  )
end

@inline Base.parent(set::SortedSet) = getfield(set, :data)
@inline order(set::SortedSet) = getfield(set, :order)

# Dictionaries.jl interface
const SortedIndices = SortedSet

# Inner constructor.
# Sorts and makes unique as needed.
function SortedSet{T,Data,Order}(
  a::Data, order::Order
) where {T,Data<:AbstractArray{T},Order<:Ordering}
  if !issorted(a, order)
    a = SmallVectors.sort(a, order)
  end
  if !alluniquesorted(a, order)
    a = uniquesorted(a, order)
  end
  return _SortedSet(a, order)
end

@inline function SortedSet{T,Data,Order}(
  a::AbstractArray, order::Ordering
) where {T,Data<:AbstractArray{T},Order<:Ordering}
  return SortedSet{T,Data,Order}(convert(Data, a), convert(Order, order))
end

@inline function SortedSet{T,Data}(
  a::AbstractArray, order::Order
) where {T,Data<:AbstractArray{T},Order<:Ordering}
  return SortedSet{T,Data,Order}(a, order)
end

@inline function SortedSet(a::Data, order::Ordering) where {T,Data<:AbstractArray{T}}
  return SortedSet{T,Data}(a, order)
end

# Accept other inputs like `Tuple`.
@inline function SortedSet(itr, order::Ordering)
  return SortedSet(collect(itr), order)
end

@inline function SortedSet{T,Data}(
  a::Data; lt=isless, by=identity, rev::Bool=false
) where {T,Data<:AbstractArray{T}}
  return SortedSet{T,Data}(a, ord(lt, by, rev))
end

# Traits
@inline SmallVectors.InsertStyle(::Type{<:SortedSet{T,Data}}) where {T,Data} = InsertStyle(
  Data
)
@inline SmallVectors.thaw(set::SortedSet) = SortedSet(thaw(parent(set)), order(set))
@inline SmallVectors.freeze(set::SortedSet) = SortedSet(freeze(parent(set)), order(set))

@propagate_inbounds SortedSet(; kwargs...) = SortedSet{Any}([]; kwargs...)
@propagate_inbounds SortedSet{T}(; kwargs...) where {T} = SortedSet{T,Vector{T}}(
  T[]; kwargs...
)
@propagate_inbounds SortedSet{T,Data}(; kwargs...) where {T,Data} = SortedSet{T}(
  Data(); kwargs...
)

@propagate_inbounds SortedSet(iter; kwargs...) = SortedSet(collect(iter); kwargs...)
@propagate_inbounds SortedSet{T}(iter; kwargs...) where {T} = SortedSet{T}(
  collect(T, iter); kwargs...
)

@propagate_inbounds SortedSet(a::AbstractArray{T}; kwargs...) where {T} = SortedSet{T}(
  a; kwargs...
)
@propagate_inbounds SortedSet{T}(a::AbstractArray{T}; kwargs...) where {T} = SortedSet{
  T,typeof(a)
}(
  a; kwargs...
)

@propagate_inbounds SortedSet{T,Data}(a::AbstractArray; kwargs...) where {T,Data<:AbstractArray{T}} = SortedSet{
  T,Data
}(
  Data(a); kwargs...
)

function Base.convert(::Type{AbstractIndices{T}}, set::SortedSet) where {T}
  return convert(SortedSet{T}, set)
end
function Base.convert(::Type{SortedSet}, set::AbstractIndices{T}) where {T}
  return convert(SortedSet{T}, set)
end
function Base.convert(::Type{SortedSet{T}}, set::AbstractIndices) where {T}
  return convert(SortedSet{T,Vector{T}}, set)
end
function Base.convert(
  ::Type{SortedSet{T,Data}}, set::AbstractIndices
) where {T,Data<:AbstractArray{T}}
  a = convert(Data, collect(T, set))
  return @inbounds SortedSet{T,typeof(a)}(a)
end

Base.convert(::Type{SortedSet{T}}, set::SortedSet{T}) where {T} = set
function Base.convert(
  ::Type{SortedSet{T}}, set::SortedSet{<:Any,Data}
) where {T,Data<:AbstractArray{T}}
  return convert(SortedSet{T,Data}, set)
end
function Base.convert(
  ::Type{SortedSet{T,Data}}, set::SortedSet{T,Data}
) where {T,Data<:AbstractArray{T}}
  return set
end
function Base.convert(
  ::Type{SortedSet{T,Data}}, set::SortedSet
) where {T,Data<:AbstractArray{T}}
  a = convert(Data, parent(set))
  return @inbounds SortedSet{T,Data}(a)
end

# Basic interface
@propagate_inbounds function Base.iterate(set::SortedSet{T}, state...) where {T}
  return iterate(parent(set), state...)
end

@inline function Base.in(i::T, set::SortedSet{T}) where {T}
  return _insorted(i, parent(set), order(set))
end
@inline Base.IteratorSize(::SortedSet) = Base.HasLength()
@inline Base.length(set::SortedSet) = length(parent(set))

function Base.:(==)(set1::SortedSet, set2::SortedSet)
  if length(set1) ≠ length(set2)
    return false
  end
  for (j1, j2) in zip(set1, set2)
    if j1 ≠ j2
      return false
    end
  end
  return true
end

function Base.issetequal(set1::SortedSet, set2::SortedSet)
  if length(set1) ≠ length(set2)
    return false
  end
  if order(set1) ≠ order(set2)
    # TODO: Make sure this actually sorts!
    set2 = SortedSet(parent(set2), order(set1))
  end
  for (j1, j2) in zip(set1, set2)
    if lt(order(set1), j1, j2) || lt(order(set1), j2, j1)
      return false
    end
  end
  return true
end

@inline Dictionaries.istokenizable(::SortedSet) = true
@inline Dictionaries.tokentype(::SortedSet) = Int
@inline Dictionaries.iteratetoken(set::SortedSet, s...) = iterate(
  LinearIndices(parent(set)), s...
)
@inline function Dictionaries.iteratetoken_reverse(set::SortedSet)
  li = LinearIndices(parent(set))
  if isempty(li)
    return nothing
  else
    t = last(li)
    return (t, t)
  end
end
@inline function Dictionaries.iteratetoken_reverse(set::SortedSet, t)
  li = LinearIndices(parent(set))
  t -= 1
  if t < first(li)
    return nothing
  else
    return (t, t)
  end
end

@inline function Dictionaries.gettoken(set::SortedSet, i)
  a = parent(set)
  r = searchsorted(a, i, order(set))
  @assert 0 ≤ length(r) ≤ 1 # If > 1, means the elements are not unique
  length(r) == 0 && return (false, 0)
  return (true, convert(Int, only(r)))
end
@propagate_inbounds Dictionaries.gettokenvalue(set::SortedSet, x::Int) = parent(set)[x]

@inline Dictionaries.isinsertable(set::SortedSet) = isinsertable(parent(set))

@inline function Dictionaries.gettoken!(set::SortedSet{T}, i::T, values=()) where {T}
  a = parent(set)
  r = searchsorted(a, i, order(set))
  @assert 0 ≤ length(r) ≤ 1 # If > 1, means the elements are not unique
  if length(r) == 0
    insert!(a, first(r), i)
    foreach(v -> resize!(v, length(v) + 1), values)
    return (false, last(LinearIndices(a)))
  end
  return (true, convert(Int, only(r)))
end

@inline function Dictionaries.deletetoken!(set::SortedSet, x::Int, values=())
  deleteat!(parent(set), x)
  foreach(v -> deleteat!(v, x), values)
  return set
end

@inline function Base.empty!(set::SortedSet, values=())
  empty!(parent(set))
  foreach(empty!, values)
  return set
end

# TODO: Make into `MSmallVector`?
# More generally, make a `thaw(::AbstractArray)` function to return
# a mutable version of an AbstractArray.
@inline Dictionaries.empty_type(::Type{SortedSet{T,D,Order}}, ::Type{T}) where {T,D,Order} = SortedSet{
  T,Dictionaries.empty_type(D, T),Order
}

@inline Dictionaries.empty_type(::Type{<:AbstractVector}, ::Type{T}) where {T} = Vector{T}

function Base.empty(set::SortedSet{T,D}, ::Type{T}) where {T,D}
  return Dictionaries.empty_type(typeof(set), T)(D(), order(set))
end

@inline function Base.copy(set::SortedSet, ::Type{T}) where {T}
  if T === eltype(set)
    SortedSet(copy(parent(set)), order(set))
  else
    SortedSet(convert(AbstractArray{T}, parent(set)), order(set))
  end
end

# TODO: Can this take advantage of sorting?
@inline function Base.filter!(pred, set::SortedSet)
  filter!(pred, parent(set))
  return set
end

function Dictionaries.randtoken(rng::Random.AbstractRNG, set::SortedSet)
  return rand(rng, keys(parent(set)))
end

@inline function Base.sort!(set::SortedSet; lt=isless, by=identity, rev::Bool=false)
  @assert Base.Sort.ord(lt, by, rev) == order(set)
  # No-op, should be sorted already.
  return set
end

# Custom faster operations (not required for interface)
function Base.union!(set::SortedSet, items::SortedSet)
  if order(set) ≠ order(items)
    # Reorder if the orderings are different.
    items = SortedSet(parent(set), order(set))
  end
  unionsortedunique!(parent(set), parent(items), order(set))
  return set
end

function Base.union(set::SortedSet, items::SortedSet)
  if order(set) ≠ order(items)
    # TODO: Reorder if the orderings are different.
    items = SortedSet(parent(set), order(set))
  end
  out = unionsortedunique(parent(set), parent(items), order(set))
  return SortedSet(out, order(set))
end

function Base.union(set::SortedSet, items)
  return union(set, SortedSet(items, order(set)))
end

function Base.intersect(set::SortedSet, items::SortedSet)
  # TODO: Make an `intersectsortedunique`.
  return intersect(NotInsertable(), set, items)
end

function Base.setdiff(set::SortedSet, items)
  return setdiff(set, SortedSet(items, order(set)))
end

function Base.setdiff(set::SortedSet, items::SortedSet)
  # TODO: Make an `setdiffsortedunique`.
  return setdiff(NotInsertable(), set, items)
end

function Base.symdiff(set::SortedSet, items)
  return symdiff(set, SortedSet(items, order(set)))
end

function Base.symdiff(set::SortedSet, items::SortedSet)
  # TODO: Make an `symdiffsortedunique`.
  return symdiff(NotInsertable(), set, items)
end
