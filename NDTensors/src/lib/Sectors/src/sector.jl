default_sector_size() = 4
default_sector_label() = Label{String7,Tuple{Int,Int},Category{String7}}
default_sector_storage() = SmallSet{default_sector_size(),default_sector_label()}

"""
A Sector is a sorted collection of named
symmetry Labels such as ("J", 1, SU(2)) or ("N", 2, U(1))
"""
struct Sector{Storage}
  storage::Storage
  global _Sector(storage) = new{typeof(storage)}(storage)
end

function Sector(v::Vector; storage_type=default_sector_storage(),
                           storage_kwargs=(;by=name))
  #LabelType = isempty(v) ? Label : typeof(first(v))
  #StorageType = 
  return _Sector(storage_type(v; storage_kwargs...))
end

function Sector(t1::Tuple, ts...; 
                label_kwargs=(;),
                kws...)
  return Sector([Label(t...;label_kwargs...) for t in (t1, ts...)]; kws...)
end

# Convenience constructor where extra parenthesis not required for
# a single label
Sector(args...; kws...) = Sector((args...,); kws...)

storage(q::Sector) = q.storage
nactive(q::Sector) = length(storage(q))
isactive(q::Sector) = (nactive(q) != 0)

#
# TODO: update code below to specialize to Sector
# types only having set behavior?
# Maybe by defining const SetSector = Sector{<:AbstractSet} ?
#

Base.union(q1::Sector, q2::Sector) = _Sector(union(storage(q1), storage(q2)))
Base.union(q1::Sector, v::Union{Vector,Tuple}) = _Sector(union(storage(q1), v))
Base.symdiff(q1::Sector, q2::Sector) = _Sector(symdiff(storage(q1), storage(q2)))
Base.setdiff(q1::Sector, q2::Sector) = _Sector(setdiff(storage(q1), storage(q2)))
Base.intersect(q1::Sector, q2::Sector) = _Sector(intersect(storage(q1), storage(q2)))
Base.iterate(q::Sector, args...) = iterate(storage(q), args...)

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
Base.:(*)(a::Sector, b::Sector) = ⊗(a, b)

# Direct sum of Sector and vectors of Sectors
⊕(a::Sector, b::Sector) = [a, b]
⊕(v::Vector{<:Sector}, b::Sector) = vcat(v, b)
⊕(a::Sector, v::Vector{Sector}) = vcat(a, v)

function Base.:(==)(A::Sector, B::Sector)
  common_labels = zip(intersect(A, B), intersect(B, A))
  common_labels_match = all(t -> (t[1] == t[2]), common_labels)
  unique_labels_zero = all(l -> iszero(val(l)), symdiff(A, B))
  return common_labels_match && unique_labels_zero
end

function Base.show(io::IO, q::Sector)
  Na = nactive(q)
  print(io, "Sector(")
  for (n, s) in enumerate(storage(q))
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

function Base.show(io::IO, q::Vector{<:Sector})
  isempty(q) && return nothing
  symbol = ""
  for l in q
    print(io, symbol, l)
    symbol = " ⊕ "
  end
end
