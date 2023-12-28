
struct Sector <: AbstractNamedSet
  data::NamedTuple
  Sector(nt::NamedTuple) = new(nt_sort(nt))
end

Sector(; kws...) = Sector((;kws...))

data(s::Sector) = s.data

Base.keys(S::Sector) = keys(data(S))
Base.values(S::Sector) = values(data(S))
Base.getindex(S::Sector,args...) = getindex(data(S),args...)

function Base.iterate(S::Sector,state=1)
  (state > length(S)) && (return nothing)
  return (keys(S)[state]=>S[state],state+1)
end

"""
  ⊗(A::Sector,B::Sector)

Fuse two Sectors producing a vector of Sectors
formed by fusing the matching sectors of A and B.
Any sectors present in B but not in A and vice versa
are treated as if they were present but had the value zero.
"""
function ⊗(A::Sector, B::Sector)
  println()
  qs = [A]
  @show qs
  for (la, lb) in zip(intersect(A, B), intersect(B, A))
    @assert la[1] == lb[1]
    fused_vals = ⊗(la[2], lb[2])
    @show fused_vals
    @show qs
    println()
    qs = [union(q, Sector(la[1]=v)) for ns in ⊗(la, lb) for q in qs]
  end
  # Include sectors of B not in A
  qs = [union(B, q) for q in qs]
  return qs
end

function Base.:(==)(A::Sector, B::Sector)
  common_labels = zip(intersect(A, B), intersect(B, A))
  common_labels_match = all(t -> (t[1] == t[2]), common_labels)
  unique_labels_zero = all(l -> istrivial(l), symdiff(A, B))
  return common_labels_match && unique_labels_zero
end

function Base.show(io::IO, sd::Sector)
  print(io, "Sector")
  print(io,data(sd))
end
