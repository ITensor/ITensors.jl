
# This is a cache of [Val(1), Val(2), ...]
# Hard-coded for now to only handle tensors up to order 100
const ValCache = Val[Val(n) for n in 0:100]
# Faster conversions of collection to tuple than `Tuple(::AbstractVector)`
_NTuple(::Val{N}, v::Vector{T}) where {N,T} = ntuple(n -> v[n], Val(N))
_Tuple(v::Vector{T}) where {T} = _NTuple(ValCache[length(v) + 1], v)
_Tuple(t::Tuple) = t

"""
    ValLength(::Type{NTuple{N}}) = Val{N}
"""
ValLength(::Type{NTuple{N,T}}) where {N,T} = Val{N}

"""
    ValLength(::NTuple{N}) = Val(N)
"""
ValLength(::NTuple{N}) where {N} = Val(N)

# Only to help with backwards compatibility, this
# is not type stable and therefore not efficient.
ValLength(v::Vector) = Val(length(v))

ValLength(::Tuple{Vararg{Any,N}}) where {N} = Val(N)

ValLength(::Type{<:Tuple{Vararg{Any,N}}}) where {N} = Val{N}

ValLength(::CartesianIndex{N}) where {N} = Val(N)
ValLength(::Type{CartesianIndex{N}}) where {N} = Val{N}

push(s::Tuple, val) = (s..., val)

pushfirst(s::Tuple, val) = (val, s...)

pop(s::NTuple{N}) where {N} = ntuple(i -> s[i], Val(N - 1))

popfirst(s::NTuple{N}) where {N} = ntuple(i -> s[i + 1], Val(N - 1))

# Permute some other type by perm
# (for example, tuple, MVector, etc.)
# as long as the constructor accepts a tuple
@inline function _permute(s, perm)
  return ntuple(i -> s[perm[i]], ValLength(perm))
end

permute(s::Tuple, perm) = _permute(s, perm)

# TODO: is this needed?
function permute(s::T, perm) where {T<:NTuple}
  return T(_permute(s, perm))
end

function permute(s::T, perm) where {T}
  return T(_permute(Tuple(s), perm))
end

# TODO: This is to handle Vector, is this correct?
permute(s::AbstractVector, perm) = _permute(s, perm)

sim(s::NTuple) = s

# type stable findfirst
@inline _findfirst(args...) = (i = findfirst(args...); i === nothing ? 0 : i)

"""
    getperm(col1,col2)

Get the permutation that takes collection 2 to collection 1,
such that col2[p].==col1
"""
@inline function getperm(s1, s2)
  return ntuple(i -> _findfirst(==(@inbounds s1[i]), s2), length(s1))
end

"""
    getperms(col1,col2,col3)

Get the permutations that takes collections 2 and 3 to collection 1.
"""
function getperms(s, s1, s2)
  N = length(s)
  N1 = length(s1)
  N2 = length(s2)
  N1 + N2 ≠ N && error("Size of partial sets don't match with total set")
  perm1 = ntuple(i -> findfirst(==(s1[i]), s), Val(N1))
  perm2 = ntuple(i -> findfirst(==(s2[i]), s), Val(N2))
  isperm((perm1..., perm2...)) ||
    error("Combined permutations are $((perm1...,perm2...)), not a valid permutation")
  return perm1, perm2
end

function invperm!(permres, perm)
  for i in 1:length(perm)
    permres[perm[i]] = i
  end
  return permres
end

function invperm(perm::NTuple{N,Int}) where {N}
  mpermres = MVector{N,Int}(undef)
  invperm!(mpermres, perm)
  return Tuple(mpermres)
end

function invperm(perm)
  permres = similar(perm)
  invperm!(permres, perm)
  return permres
end

# Override TupleTools.isperm to speed up
# Strided.permutedims a bit (see:
# https://github.com/Jutho/Strided.jl/issues/15)
function isperm(p::NTuple{N}) where {N}
  N < 6 && return Base.isperm(p)
  used = @MVector zeros(Bool, N)
  for a in p
    (0 < a <= N) && (used[a] ⊻= true) || return false
  end
  return true
end

"""
    is_trivial_permutation(P)

Determine if P is a trivial permutation.
"""
function is_trivial_permutation(P)
  #isperm(P) || error("Input is not a permutation")
  # TODO: use `all(n->P[n]==n,1:length(P))`?
  N = length(P)
  for n in 1:N
    @inbounds P[n] != n && return false
  end
  return true
end

# Combine a bunch of tuples
@inline flatten(x) = x
@inline flatten(x, y) = (x..., y...)
@inline flatten(x, y, z...) = (x..., flatten(y, z...)...)

function _deleteat(t, pos, i)
  i < pos && return t[i]
  return t[i + 1]
end

function deleteat(t::NTuple{N}, pos::Integer) where {N}
  return ntuple(i -> _deleteat(t, pos, i), Val(N - 1))
end

deleteat(t::Tuple, I::Tuple{Int}) = deleteat(t, I[1])
function deleteat(t::Tuple, I::Tuple{Int,Int,Vararg{Int}})
  return deleteat_sorted(t, sort(I; rev=true))
end

deleteat_sorted(t::Tuple, pos::Int64) = deleteat(t, pos[1])
deleteat_sorted(t::Tuple, pos::Tuple{Int}) = deleteat(t, pos[1])
function deleteat_sorted(t::Tuple, pos::NTuple{N,Int}) where {N}
  return deleteat_sorted(deleteat_sorted(t, pos[1]), Base.tail(pos))
end

# Make a slice of the block on the specified dimensions
# Make this a generic tupletools function (TupleTools.jl calls it getindices)
function getindices(t::Tuple, I::NTuple{N,Int}) where {N}
  return ntuple(i -> t[I[i]], Val(N))
end

function _insertat(t, pos, n_insert, val, i)
  if i < pos
    return t[i]
  elseif i > pos + n_insert - 1
    return t[i - n_insert + 1]
  end
  return val[i - pos + 1]
end

"""
    insertat

Remove the value at pos and insert the elements in val
"""
function insertat(t::NTuple{N}, val::NTuple{M}, pos::Integer) where {N,M}
  return ntuple(i -> _insertat(t, pos, M, val, i), Val(N + M - 1))
end

function insertat(t::NTuple{N}, val, pos::Integer) where {N}
  return insertat(t, tuple(val), pos)
end

function _insertafter(t, pos, n_insert, val, i)
  if i <= pos
    return t[i]
  elseif i > pos + n_insert
    return t[i - n_insert]
  end
  return val[i - pos]
end

"""
    insertafter(t, val, pos)

Insert the elements in val after the position pos
"""
function insertafter(t::NTuple{N}, val::NTuple{M}, pos::Integer) where {N,M}
  return ntuple(i -> _insertafter(t, pos, M, val, i), Val(N + M))
end

function insertafter(t::NTuple{N}, val, pos::Integer) where {N}
  return insertafter(t, tuple(val), pos)
end

"""
    isdisjoint(s1, s2)

Determine if s1 and s2 have no overlapping elements.
"""
function isdisjoint(s1, s2)
  for i1 in 1:length(s1)
    for i2 in 1:length(s2)
      s1[i1] == s2[i2] && return false
    end
  end
  return true
end

"""
    diff(t::Tuple)

For a tuple of length N, return a tuple of length N-1
where element i is t[i+1] - t[i].
"""
diff(t::NTuple{N}) where {N} = ntuple(i -> t[i + 1] - t[i], Val(N - 1))

function count_unique(labelsT1, labelsT2)
  count = 0
  for l1 in labelsT1
    l1 ∉ labelsT2 && (count += 1)
  end
  return count
end

function count_common(labelsT1, labelsT2)
  count = 0
  for l1 in labelsT1
    l1 ∈ labelsT2 && (count += 1)
  end
  return count
end

function intersect_positions(labelsT1, labelsT2)
  for i1 in 1:length(labelsT1)
    for i2 in 1:length(labelsT2)
      if labelsT1[i1] == labelsT2[i2]
        return i1, i2
      end
    end
  end
  return nothing
end

function is_replacement(labelsT1, labelsT2)
  return count_unique(labelsT1, labelsT2) == 1 && count_common(labelsT1, labelsT2) == 1
end

function is_combiner(labelsT1, labelsT2)
  return count_unique(labelsT1, labelsT2) == 1 && count_common(labelsT1, labelsT2) > 1
end

function is_uncombiner(labelsT1, labelsT2)
  return count_unique(labelsT1, labelsT2) > 1 && count_common(labelsT1, labelsT2) == 1
end
