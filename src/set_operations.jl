
#
# Set operations
# These are custom implementations of the set operations in
# abstractset.jl. However, they do not convert input collections
# to Sets. Therefore, they have higher complexity (O(N²) instead
# of O(N) for N elements) but they are faster for small sets.
# In addition, they assume that elements of the input collections
# are already unique, i.e. that the inputs are "set-like".
# Therefore, you should call a function like `unique` before inputting
# if you're not sure the collections themselves will be unique.
#

# A version of Base.setdiff that scales quadratically in the number of elements
# and assumes the elements of each input set are already unique.
_setdiff(s) = Base.copymutable(s)
_setdiff(s, itrs...) = _setdiff!(Base.copymutable(s), itrs...)
function _setdiff!(s, itrs...)
  for x in itrs
    _setdiff!(s, x)
  end
  return s
end
function _setdiff!(s, itr)
  isempty(s) && return s
  for x in itr
    n = findfirst(==(x), s)
    !isnothing(n) && deleteat!(s, n)
  end
  return s
end

# A version of Base.intersect that scales quadratically in the number of elements
# and assumes the elements of each input set are already unique.
_intersect(s) = Base.copymutable(s)
_intersect(s, itr, itrs...) = _intersect!(_intersect(s, itr), itrs...)
# XXX: Base has `s` and `itr` swapped in the definition, which one is correct?
# Is this special case needed, or is `filter!` sufficient?
_intersect(s, itr) = Base.mapfilter(in(itr), push!, s, Base.emptymutable(s, eltype(s)))
function _intersect!(s, itrs...)
  for x in itrs
    _intersect!(s, x)
  end
  return s
end
_intersect!(s, s2) = filter!(in(s2), s)

# A version of Base.symdiff that scales quadratically in the number of elements
# and assumes the elements of each input set are already unique.
_symdiff(s) = Base.copymutable(s)
function _symdiff(s, itrs...)
  return _symdiff!(Base.emptymutable(s, Base.promote_eltype(s, itrs...)), s, itrs...)
end
function _symdiff!(s, itrs...)
  for x in itrs
    _symdiff!(s, x)
  end
  return s
end
function _symdiff!(s, itr)
  if isempty(s)
    append!(s, itr)
    return s
  end
  for x in itr
    n = findfirst(==(x), s)
    !isnothing(n) ? deleteat!(s, n) : push!(s, x)
  end
  return s
end

# A version of Base.union that scales quadratically in the number of elements
# and assumes the elements of each input set are already unique.
_union(s) = Base.copymutable(s)
function _union(s, sets...)
  return _union!(Base.emptymutable(s, Base.promote_eltype(s, sets...)), s, sets...)
end
function _union!(s, sets...)
  for x in sets
    _union!(s, x)
  end
  return s
end
function _union!(s, itr)
  Base.haslength(itr) && sizehint!(s, length(s) + Int(length(itr))::Int)
  for x in itr
    x ∉ s && push!(s, x)
  end
  return s
end
