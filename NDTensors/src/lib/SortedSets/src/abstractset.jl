abstract type AbstractSet{T} <: AbstractIndices{T} end

# Specialized versions of set operations for `AbstractSet`
# that allow more specialization.

function Base.union(i::AbstractSet, itr)
  return union(InsertStyle(i), i, itr)
end

function Base.union(::InsertStyle, i::AbstractSet, itr)
  return error("Not implemented")
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

function Base.intersect(i::AbstractSet, itr)
  return intersect(InsertStyle(i), i, itr)
end

function Base.intersect(::InsertStyle, i::AbstractSet, itr)
  return error("Not implemented")
end

function Base.intersect(::IsInsertable, i::AbstractSet, itr)
  out = copy(i)
  intersect!(out, itr)
  return out
end

function Base.intersect(::NotInsertable, i::AbstractSet, itr)
  out = empty(i)
  union!(out, i)
  intersect!(out, itr)
  return out
end

function Base.setdiff(i::AbstractSet, itr)
  return setdiff(InsertStyle(i), i, itr)
end

function Base.setdiff(::InsertStyle, i::AbstractSet, itr)
  return error("Not implemented")
end

function Base.setdiff(::IsInsertable, i::AbstractSet, itr)
  out = copy(i)
  setdiff!(out, itr)
  return out
end

function Base.setdiff(::NotInsertable, i::AbstractSet, itr)
  out = empty(i)
  union!(out, i)
  setdiff!(out, itr)
  return out
end

function Base.symdiff(i::AbstractSet, itr)
  return symdiff(InsertStyle(i), i, itr)
end

function Base.symdiff(::InsertStyle, i::AbstractSet, itr)
  return error("Not implemented")
end

function Base.symdiff(::IsInsertable, i::AbstractSet, itr)
  out = copy(i)
  symdiff!(out, itr)
  return out
end

function Base.symdiff(::NotInsertable, i::AbstractSet, itr)
  out = empty(i)
  union!(out, i)
  symdiff!(out, itr)
  return out
end
