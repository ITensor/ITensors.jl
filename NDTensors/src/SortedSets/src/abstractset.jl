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
