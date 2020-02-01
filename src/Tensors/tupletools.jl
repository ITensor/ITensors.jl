export insertat,
       insertafter,
       deleteat,
       getindices,
       tuplecat,
       getperm,
       getperms,
       invperm,
       permute,
       ValLength,
       is_trivial_permutation,
       sim,
       isdisjoint,
       count_common,
       count_unique

"""
ValLength(::Type{NTuple{N}}) = Val{N}
"""
ValLength(::Type{NTuple{N,T}}) where {N,T} = Val{N}

"""
ValLength(::NTuple{N}) = Val(N)
"""
ValLength(::NTuple{N}) where {N} = Val(N)

ValLength(::CartesianIndex{N}) where {N} = Val(N)
ValLength(::Type{CartesianIndex{N}}) where {N} = Val{N}

# Permute some other type by perm
# (for example, tuple, MVector, etc.)
# as long as the constructor accepts a tuple
function permute(s::T,perm) where {T}
  return T(ntuple(i->s[perm[i]], ValLength(T)()))
end

sim(s::NTuple) = s

"""
getperm(col1,col2)

Get the permutation that takes collection 2 to collection 1,
such that col2[p].==col1
"""
function getperm(s1,s2)
  return ntuple(i->findfirst(==(s1[i]),s2),Val(ndims(s1)))
end

"""
getperm(col1,col2,col3)

Get the permutations that takes collections 2 and 3 to collection 1.
"""
function getperms(s,s1,s2)
  N = ndims(s)
  N1 = ndims(s1)
  N2 = ndims(s2)
  N1+N2≠N && error("Size of partial sets don't match with total set")
  perm1 = ntuple(i->findfirst(==(s1[i]),s),Val(N1))
  perm2 = ntuple(i->findfirst(==(s2[i]),s),Val(N2))
  isperm((perm1...,perm2...)) || error("Combined permutations are $((perm1...,perm2...)), not a valid permutation")
  return perm1,perm2
end

function invperm!(permres,perm)
  for i in 1:length(perm)
    permres[perm[i]] = i
  end
  return permres
end

function invperm(perm::NTuple{N,Int}) where {N}
  mpermres = MVector{N,Int}(undef)
  invperm!(mpermres,perm)
  return Tuple(mpermres)
end

function invperm(perm)
  permres = similar(perm)
  invperm!(permres,perm)
  return permres
end

"""
Determine if P is a trivial permutation. Errors if P is not a valid
permutation.
"""
function is_trivial_permutation(P)
  isperm(P) || error("Input is not a permutation")
  # TODO: use `all(n->P[n]==n,1:length(P))`?
  N = length(P)
  for n = 1:N
    P[n]!=n && return false
  end
  return true
end

# Combine a bunch of tuples
# TODO: move this functionality to IndexSet, combine with unioninds?
@inline tuplecat(x) = x
@inline tuplecat(x, y) = (x..., y...)
@inline tuplecat(x, y, z...) = (x..., tuplecat(y, z...)...)

#function tuplecat(is1::NTuple{N1},
#                  is2::NTuple{N2}) where {N1,N2}
#  return tuple(is1...,is2...)
#end

function _deleteat(t,pos,i)
  i < pos && return t[i]
  return t[i+1]
end

function StaticArrays.deleteat(t::NTuple{N},pos::Integer) where {N}
  return ntuple(i -> _deleteat(t,pos,i),Val(N-1))
end

StaticArrays.deleteat(t::Tuple, I::Tuple{Int}) = deleteat(t, I[1])
function StaticArrays.deleteat(t::Tuple, I::Tuple{Int, Int, Vararg{Int}})
    return deleteat_sorted(t, sort(I, rev = true))
end

deleteat_sorted(t::Tuple,pos::Int64) = deleteat(t,pos[1])
deleteat_sorted(t::Tuple,pos::Tuple{Int}) = deleteat(t,pos[1])
deleteat_sorted(t::Tuple,pos::NTuple{N,Int}) where {N} = deleteat_sorted(deleteat_sorted(t,pos[1]),Base.tail(pos))

# Make a slice of the block on the specified dimensions
# Make this a generic tupletools function (TupleTools.jl calls it getindices)
function getindices(t::Tuple,
                    I::NTuple{N,Int}) where {N}
  return ntuple(i->t[I[i]],Val(N))
end

# Taken from TupleTools.jl
"""
    sort(t::Tuple; lt=isless, by=identity, rev::Bool=false) -> ::Tuple
Sorts the tuple `t`.
"""
Base.sort(t::Tuple; lt=isless, by=identity, rev::Bool=false) = _sort(t, lt, by, rev)
@inline function _sort(t::Tuple, lt=isless, by=identity, rev::Bool=false)
    t1, t2 = _split(t)
    t1s = _sort(t1, lt, by, rev)
    t2s = _sort(t2, lt, by, rev)
    return _merge(t1s, t2s, lt, by, rev)
end
_sort(t::Tuple{Any}, lt=isless, by=identity, rev::Bool=false) = t
_sort(t::Tuple{}, lt=isless, by=identity, rev::Bool=false) = t

function _split(t::NTuple{N}) where N
    M = N>>1
    ntuple(i->t[i], M), ntuple(i->t[i+M], N-M)
end

function _merge(t1::Tuple, t2::Tuple, lt, by, rev)
    if lt(by(first(t1)), by(first(t2))) != rev
        return (first(t1), _merge(Base.tail(t1), t2, lt, by, rev)...)
    else
        return (first(t2), _merge(t1, Base.tail(t2), lt, by, rev)...)
    end
end
_merge(t1::Tuple{}, t2::Tuple, lt, by, rev) = t2
_merge(t1::Tuple, t2::Tuple{}, lt, by, rev) = t1

#function tail(t::NTuple{N})
#  return ntuple(i -> t[i+1],Val(N-1))
#end

function _insertat(t,pos,n_insert,val,i)
  if i < pos
    return t[i]
  elseif i > pos+n_insert-1
    return t[i-n_insert+1]
  end
  return val[i-pos+1]
end

"""
insertat

Remove the value at pos and insert the elements in val
"""
function insertat(t::NTuple{N},
                  val::NTuple{M},
                  pos::Integer) where {N,M}
  return ntuple(i -> _insertat(t,pos,M,val,i),Val(N+M-1))
end

function insertat(t::NTuple{N}, val, pos::Integer) where {N}
  return insertat(t,tuple(val),pos)
end

function _insertafter(t,pos,n_insert,val,i)
  if i <= pos
    return t[i]
  elseif i > pos+n_insert
    return t[i-n_insert]
  end
  return val[i-pos]
end

"""
insertafter

Insert the elements in val after the position pos
"""
function insertafter(t::NTuple{N},
                     val::NTuple{M},
                     pos::Integer) where {N,M}
  return ntuple(i -> _insertafter(t,pos,M,val,i),Val(N+M))
end

function insertafter(t::NTuple{N}, val, pos::Integer) where {N}
  return insertafter(t,tuple(val),pos)
end

"""
Determine if s1 and s2 have no overlapping elements.
"""
function isdisjoint(s1,s2)
  for i1 ∈ 1:length(s1)
    for i2 ∈ 1:length(s2)
      s1[i1] == s2[i2] && return false
    end
  end
  return true
end

function count_unique(labelsT1,labelsT2)
  count = 0
  for l1 ∈ labelsT1
    l1 ∉ labelsT2 && (count += 1)
  end
  return count
end

function count_common(labelsT1,labelsT2)
  count = 0
  for l1 ∈ labelsT1
    l1 ∈ labelsT2 && (count += 1)
  end
  return count
end

function intersect_positions(labelsT1,labelsT2)
  for i1 = 1:length(labelsT1)
    for i2 = 1:length(labelsT2)
      if labelsT1[i1] == labelsT2[i2]
        return i1,i2
      end
    end
  end
  return nothing
end

function is_replacement(labelsT1,labelsT2)
  return count_unique(labelsT1,labelsT2) == 1 &&
         count_common(labelsT1,labelsT2) == 1
end

function is_combiner(labelsT1,labelsT2)
  return count_unique(labelsT1,labelsT2) == 1 &&
         count_common(labelsT1,labelsT2) > 1
end

function is_uncombiner(labelsT1,labelsT2)
  return count_unique(labelsT1,labelsT2) > 1 &&
         count_common(labelsT1,labelsT2) == 1
end

