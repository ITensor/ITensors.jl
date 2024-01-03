
struct Sector{Data} <: AbstractCategory
  data::Data
end

Sector(nt::NamedTuple) = Sector{NamedTuple}(nt_sort(nt))

Sector(; kws...) = Sector((; kws...))

function Sector(pairs::Pair...)
  keys = ntuple(n -> Symbol(pairs[n][1]), length(pairs))
  vals = ntuple(n -> pairs[n][2], length(pairs))
  return Sector(NamedTuple{keys}(vals))
end

Sector(v::Vector{<:AbstractCategory}) = Sector{Vector}(v)

Sector(cats::AbstractCategory...) = Sector([cats...])

data(s::Sector) = s.data

Base.isempty(S::Sector) = isempty(data(S))
Base.length(S::Sector) = length(data(S))
Base.getindex(S::Sector, args...) = getindex(data(S), args...)

#
# Set-like interface
#

const NamedSector = Sector{<:NamedTuple}

Base.intersect(s1::NamedSector, s2::NamedSector) = Sector(nt_intersect(data(s1), data(s2)))
Base.symdiff(s1::NamedSector, s2::NamedSector) = Sector(nt_symdiff(data(s1), data(s2)))
Base.union(s1::NamedSector, s2::NamedSector) = Sector(nt_union(data(s1), data(s2)))

×(nt1::NamedTuple, nt2::NamedTuple) = Sector(nt_union(nt1,nt2))
×(s1::NamedSector, c2::NamedTuple) = Sector(nt_union(data(s1), c2))
×(c1::NamedTuple, s2::NamedSector) = Sector(nt_union(c1, data(s2)))

const NamedCategory = Pair{<:Any,<:AbstractCategory}
×(c1::NamedCategory, c2::NamedCategory) = Sector(nt_union(data(Sector(c1)),data(Sector(c2))))
×(s1::NamedSector, c2::NamedCategory) = Sector(nt_union(data(s1), data(Sector(c2))))
×(c1::NamedCategory, s2::NamedSector) = Sector(nt_union(data(Sector(c1)), data(s2)))

#
# Dictionary-like interface
#

Base.keys(S::NamedSector) = keys(data(S))
Base.values(S::NamedSector) = values(data(S))

function Base.iterate(s::NamedSector, state=1)
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
function ⊗(A::NamedSector, B::NamedSector)
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

function Base.:(==)(A::NamedSector, B::NamedSector)
  common_labels = zip(intersect(A, B), intersect(B, A))
  common_labels_match = all(nl -> (nl[1] == nl[2]), common_labels)
  unique_labels_zero = all(l -> istrivial(l[2]), symdiff(A, B))
  return common_labels_match && unique_labels_zero
end

# TODO: make printing more similar to ordered case?
#       perhaps using × operator
Base.show(io::IO, s::NamedSector) = print(io, "Sector", isempty(s) ? "()" : data(s))

#
# Ordered interface
#

const OrderedSector = Sector{<:Vector}

×(c1::AbstractCategory, c2::AbstractCategory) = Sector([c1, c2])
×(s1::OrderedSector, c2::AbstractCategory) = Sector(vcat(data(s1), c2))
×(c1::AbstractCategory, s2::OrderedSector) = Sector(vcat(c1, data(s2)))

Base.:(==)(o1::OrderedSector, o2::OrderedSector) = (data(o1) == data(o2))

# Helper function for ⊗
function replace(o::OrderedSector, n::Int, val)
  d = copy(data(o))
  d[n] = val
  return Sector(d)
end

function ⊗(o1::OrderedSector, o2::OrderedSector)
  N = length(o1)
  length(o2) == N || throw(DimensionMismatch("Ordered Sectors must have same size in ⊗"))
  os = [o1]
  for n in 1:N
    os = [replace(o, n, f) for f in ⊗(o1[n], o2[n]) for o in os]
  end
  return os
end

function Base.show(io::IO, os::OrderedSector)
  isempty(os) && return nothing
  print(io, "(")
  symbol = ""
  for l in data(os)
    print(io, symbol, l)
    symbol = " × "
  end
  return print(io, ")")
end
