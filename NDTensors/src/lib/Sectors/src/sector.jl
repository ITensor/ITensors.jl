
struct Sector{Data} <: AbstractCategory
  data::Data
  global _Sector(d) = new{typeof(d)}(d)
end

Sector(nt::NamedTuple) = _Sector(nt_sort(nt))

Sector(; kws...) = Sector((; kws...))

function Sector(pairs::Pair...)
  N = length(pairs)
  keys = ntuple(n -> Symbol(pairs[n][1]), Val(N))
  vals = ntuple(n -> pairs[n][2], Val(N))
  return Sector(NamedTuple{keys}(vals))
end

data(s::Sector) = s.data

Base.isempty(S::Sector) = isempty(data(S))
Base.length(S::Sector) = length(data(S))
Base.getindex(S::Sector, args...) = getindex(data(S), args...)

#
# Set-like interface
#

Base.intersect(s1::Sector, s2::Sector) = Sector(nt_intersect(data(s1), data(s2)))
Base.symdiff(s1::Sector, s2::Sector) = Sector(nt_symdiff(data(s1), data(s2)))
Base.union(s1::Sector, s2::Sector) = Sector(nt_union(data(s1), data(s2)))

#
# Dictionary-like interface
#

Base.keys(S::Sector{<:NamedTuple}) = keys(data(S))
Base.values(S::Sector{<:NamedTuple}) = values(data(S))

function Base.iterate(s::Sector{<:NamedTuple}, state=1)
  (state > length(s)) && (return nothing)
  return (keys(s)[state] => s[state], state + 1)
end

"""
  ⊗(A::Sector,B::Sector)

Fuse two Sectors producing a vector of Sectors
formed by fusing the matching sectors of A and B.
Any sectors present in B but not in A and vice versa
are treated as if they were present but had the value zero.
"""
function ⊗(A::Sector{<:NamedTuple}, B::Sector{<:NamedTuple})
  qs = [A]
  for (la, lb) in zip(intersect(A, B), intersect(B, A))
    @assert la[1] == lb[1]
    fused_vals = ⊗(la[2], lb[2])
    qs = [union(Sector(la[1] => v), q) for v in fused_vals for q in qs]
  end
  # Include sectors of B not in A
  qs = [union(q, B) for q in qs]
  return qs
end

function Base.:(==)(A::Sector{<:NamedTuple}, B::Sector{<:NamedTuple})
  common_labels = zip(intersect(A, B), intersect(B, A))
  common_labels_match = all(nl -> (nl[1] == nl[2]), common_labels)
  unique_labels_zero = all(l -> istrivial(l[2]), symdiff(A, B))
  return common_labels_match && unique_labels_zero
end

# TODO: make printing more similar to ordered case, 
#       perhaps using × operator
Base.show(io::IO, s::Sector) = print(io, "Sector", isempty(s) ? "()" : data(s))

#
# Ordered interface
#

Sector(v::Vector{<:AbstractCategory}) = _Sector(v)

×(c1::AbstractCategory, c2::AbstractCategory) = Sector([c1, c2])
×(s1::Sector{<:Vector}, c2::AbstractCategory) = Sector(vcat(data(s1), c2))
×(c1::AbstractCategory, s2::Sector{<:Vector}) = Sector(vcat(c1, data(s2)))

Base.:(==)(o1::Sector{<:Vector}, o2::Sector{<:Vector}) = (data(o1) == data(o2))

# Helper function for ⊗
function replace(o::Sector{<:Vector}, n::Int, val)
  d = copy(data(o))
  d[n] = val
  return Sector(d)
end

function ⊗(o1::Sector{<:Vector}, o2::Sector{<:Vector})
  N = length(o1)
  length(o2) == N || throw(DimensionMismatch("Ordered Sectors must have same size in ⊗"))
  os = [o1]
  for n in 1:N
    os = [replace(o, n, f) for f in ⊗(o1[n], o2[n]) for o in os]
  end
  return os
end

function Base.show(io::IO, os::Sector{<:Vector})
  isempty(os) && return nothing
  print(io, "(")
  symbol = ""
  for l in data(os)
    print(io, symbol, l)
    symbol = " × "
  end
  return print(io, ")")
end
