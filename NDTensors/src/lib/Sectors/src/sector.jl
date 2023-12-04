const default_sector_size = 4
default_sector_data() = Label
const SStorage = SmallSet{default_sector_size,default_sector_data()}
const MSStorage = MSmallSet{default_sector_size,default_sector_data()}

"""
A Sector is a sorted collection of named
symmetry Labels such as ("J", 1, SU(2)) or ("N", 2, U(1))
"""
struct Sector{T,D<:AbstractSmallSet{T}}
  data::D
end

function Sector()
  return Sector{default_sector_data(),SStorage}(SStorage(default_sector_data()[]; by=name))
end

Sector{T}(t1::Tuple, ts...) where {T} = Sector([T(t...) for t in (t1, ts...)])

Sector(t1::Tuple, ts...) = Sector{default_sector_data()}(t1, ts...)

Sector(v::Vector) = Sector(SStorage(v; by=name))

# Convenience constructor where extra parenthesis not needed for one label:
Sector(name::String, val::Union{Number,String}, cat=U(1)) = Sector((name, val, cat))
function Sector(
  name::String, val1::Union{Number,String}, val2::Union{Number,String}, cat=U(1)
)
  return Sector((name, (val1, val2), cat))
end

# Convenience constructor where name nor extra parenthesis not needed for one label:
Sector(val::Union{Number,String}, cat=U(1)) = Sector(("", val, cat))
function Sector(val1::Union{Number,String}, val2::Union{Number,String}, cat=U(1))
  return Sector(("", (val1, val2), cat))
end

data(q::Sector) = q.data
nactive(q::Sector) = length(data(q))
isactive(q::Sector) = (nactive(q) != 0)

union(q1::Sector, q2::Sector) = Sector(union(data(q1), data(q2)))
union(q1::Sector, v::Union{Vector,Tuple}) = Sector(union(data(q1), v))
symdiff(q1::Sector, q2::Sector) = Sector(symdiff(data(q1), data(q2)))
setdiff(q1::Sector, q2::Sector) = Sector(setdiff(data(q1), data(q2)))
intersect(q1::Sector, q2::Sector) = Sector(intersect(data(q1), data(q2)))
iterate(q::Sector, args...) = iterate(data(q), args...)

"""
  ⊗(A::Sector,B::Sector)

Fuse two Sectors producing a vector of Sectors
formed by fusing the matching sectors of A and B.
Any sectors present in B but not in A and vice versa 
are treated as if they were present but had the value zero.
"""
function ⊗(A::Sector, B::Sector)
  qs = [A]
  for (la, lb) in zip(intersect(A, B), intersect(B, A))
    qs = [union(q, [ns]) for ns in ⊗(la, lb) for q in qs]
  end
  # Include sectors of B not in A
  qs = [union(B, q) for q in qs]
  return qs
end
*(a::Sector, b::Sector) = ⊗(a, b)

# Direct sum of Sector and vectors of Sectors
⊕(a::Sector, b::Sector) = [a, b]
⊕(v::Vector{<:Sector}, b::Sector) = vcat(v, b)
⊕(a::Sector, v::Vector{Sector}) = vcat(a, v)

function ==(A::Sector, B::Sector)
  common_labels = zip(intersect(A, B), intersect(B, A))
  common_labels_match = all(t -> (t[1] == t[2]), common_labels)
  unique_labels_zero = all(l -> iszero(val(l)), symdiff(A, B))
  return common_labels_match && unique_labels_zero
end

# Treat [q] as q for comparison purposes
# so we can check things like a ⊗ b = c
# when a ⊗ b returns a Vector
==(q::Sector, v::Vector{<:Sector}) = (length(v) == 1 && q == first(v))
==(v::Vector{<:Sector}, q::Sector) = (q == v)

function show(io::IO, q::Sector)
  Na = nactive(q)
  print(io, "Sector(")
  for (n, s) in enumerate(data(q))
    n > 1 && print(io, ",")
    Na > 1 && print(io, "(")
    if name(s) != ""
      print(io, "\"$(name(s))\",")
    end
    print(io, "$(val_to_str(s))")
    print(io, ",$(category(s))")
    Na > 1 && print(io, ")")
  end
  return print(io, ")")
end

function show(io::IO, q::Vector{<:Sector})
  isempty(q) && return nothing
  symbol = ""
  for l in q
    print(io, symbol, l)
    symbol = " ⊕ "
  end
end
