## # Trait determining the style of inserting into a structure
## abstract type InsertStyle end
## struct IsInsertable <: InsertStyle end
## struct NotInsertable <: InsertStyle end
## struct FastCopy <: InsertStyle end

## # Assume is insertable
## @inline InsertStyle(::Type) = IsInsertable()
## @inline InsertStyle(x) = InsertStyle(typeof(x))

## thaw(x) = error("Not implemented")
## freeze(x) = error("Not implemented")

## # Customize `isinsertable` based on the data type of
## # the `SortedSet`.
## @inline isinsertable(::AbstractArray) = true

## # TODO: Organize into AbstractIndices file.
## insert(inds::AbstractIndices, i) = insert!(copy(inds), i)
## delete(inds::AbstractIndices, i) = delete!(copy(inds), i)

## # DictionariesSmallVectorsExt
## @inline isinsertable(::AbstractSmallVector) = true
## @inline isinsertable(::SmallVector) = false
## @inline Dictionaries.empty_type(::Type{SmallVector{S,T}}, ::Type{T}) where {S,T} = MSmallVector{S,T}
## @inline InsertStyle(::Type{<:SmallVector}) = FastCopy()

## @inline thaw(vec::SmallVector) = MSmallVector(vec)
## @inline freeze(vec::SmallVector) = vec
## @inline thaw(vec::MSmallVector) = copy(vec)
## @inline freeze(vec::MSmallVector) = SmallVector(vec)

abstract type AbstractSet{T} <: AbstractIndices{T} end

# Specialized versions of set operations for `AbstractSet`
# that allow more specialization.

function Base.union(i::AbstractSet, itr)
  return union(InsertStyle(i), i, itr)
end

function Base.union(::IsInsertable, i::AbstractSet, itr)
  out = copy(i)
  union!(out, itr)
  return out
end

function Base.union(::NotInsertable, i::AbstractSet, itr)
  out = empty(i)
  union!(out, i)
  union!(out, itr)
  return out
end

function Base.union(::FastCopy, i::AbstractSet, itr)
  error("Not implemented")
  out = thaw(i)
  union!(out, itr)
  return freeze(out)
end

## # Dictionaries.jl version
## function Base.union(i::AbstractSet{T}, itr) where {T}
##   if Base.IteratorEltype(itr) === Base.EltypeUnknown()
##     itr = collect(itr)
##   end
##   T2 = eltype(itr)
##   Tout = promote_type(T, T2)
##   if isinsertable(i)
##     out = copy(i, Tout)
##     union!(out, itr)
##   else
##     out = empty(i, Tout)
##     union!(out, i)
##     union!(out, itr)
##   end
##   return out
## end

function Base.intersect(i::AbstractSet, itr)
  error("Not implemented")
  if isinsertable(i)
    out = copy(i)
    intersect!(out, itr)
  else
    out = empty(i)
    union!(out, i)
    intersect!(out, itr)
  end
  return out
end

function Base.setdiff(i::AbstractSet, itr)
  error("Not implemented")
  if isinsertable(i)
    out = copy(i)
    setdiff!(out, itr)
  else
    out = empty(i)
    union!(out, i)
    setdiff!(out, itr)
  end
  return out
end

function Base.symdiff(i::AbstractSet{T}, itr) where {T}
  error("Not implemented")
  if Base.IteratorEltype(itr) === Base.EltypeUnknown()
    itr = collect(itr)
  end
  T2 = eltype(itr)
  Tout = promote_type(T, T2)

  if isinsertable(i)
    out = copy(i, Tout)
    symdiff!(out, itr)
  else
    out = empty(i, Tout)
    union!(out, i)
    symdiff!(out, itr)
  end
  return out
end

## # AbstractWrappedSet: a wrapper around an `AbstractIndices`
## # with methods automatically forwarded via `parent`.
## abstract type AbstractWrappedSet{T,D} <: AbstractSet{T} end
## 
## # Required interface
## Base.parent(tags::AbstractWrappedSet) = error("Not implemented")
## Dictionaries.empty_type(::Type{AbstractWrappedSet{I}}, ::Type{I}) where {I} = error("Not implemented")
## thaw(::AbstractWrappedSet) = error("Not implemented")
## freeze(::AbstractWrappedSet) = error("Not implemented")
## rewrap(::AbstractWrappedSet, data) = error("Not implemented")
## 
## # Traits
## @inline InsertStyle(::Type{<:AbstractWrappedSet{T,D}}) where {T,D} = InsertStyle(D)
## 
## # AbstractIndices interface
## @propagate_inbounds function Base.iterate(tags::AbstractWrappedSet, state...)
##   return iterate(parent(tags), state...)
## end
## 
## # `I` is needed to avoid ambiguity error.
## @inline Base.in(tag::I, tags::AbstractWrappedSet{I}) where {I} = in(tag, parent(tags))
## @inline Base.IteratorSize(tags::AbstractWrappedSet) = Base.IteratorSize(parent(tags))
## @inline Base.length(tags::AbstractWrappedSet) = length(parent(tags))
## 
## @inline Dictionaries.istokenizable(tags::AbstractWrappedSet) = istokenizable(parent(tags))
## @inline Dictionaries.tokentype(tags::AbstractWrappedSet) = tokentype(parent(tags))
## @inline Dictionaries.iteratetoken(inds::AbstractWrappedSet, s...) = iterate(parent(inds), s...)
## @inline function Dictionaries.iteratetoken_reverse(inds::AbstractWrappedSet)
##   return iteratetoken_reverse(parent(inds))
## end
## @inline function Dictionaries.iteratetoken_reverse(inds::AbstractWrappedSet, t)
##   return iteratetoken_reverse(parent(inds), t)
## end
## 
## @inline function Dictionaries.gettoken(inds::AbstractWrappedSet, i)
##   return gettoken(parent(inds), i)
## end
## @propagate_inbounds Dictionaries.gettokenvalue(inds::AbstractWrappedSet, x) =
##   gettokenvalue(parent(inds), x)
## 
## @inline Dictionaries.isinsertable(tags::AbstractWrappedSet) = isinsertable(parent(tags))
## 
## # Specify `I` to fix ambiguity error.
## @inline function Dictionaries.gettoken!(tags::AbstractWrappedSet{I}, i::I, values=()) where {I}
##   return gettoken!(parent(tags), i, values)
## end
## 
## @inline function Dictionaries.deletetoken!(tags::AbstractWrappedSet, x, values=())
##   deletetoken!(parent(tags), x, values)
##   return tags
## end
## 
## @inline function Base.empty!(inds::AbstractWrappedSet, values=())
##   empty!(parent(inds))
##   return inds
## end
## 
## # Not defined to be part of the `AbstractIndices` interface,
## # but seems to be needed.
## @inline function Base.filter!(pred, inds::AbstractWrappedSet)
##   filter!(pred, parent(inds))
##   return inds
## end
## 
## # TODO: Maybe require an implementation?
## @inline function Base.copy(tags::AbstractWrappedSet, eltype::Type)
##   return typeof(tags)(copy(parent(tags), eltype))
## end
## 
## # Not required for AbstractIndices interface but
## # helps with faster code paths
## @inline Base.union(set::AbstractWrappedSet, items) = rewrap(set, union(parent(set), items))
