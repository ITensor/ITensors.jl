
struct Sector <: AbstractNamedSet
  data::NamedTuple
  Sector(nt::NamedTuple) = new(nt_sort(nt))
end

Sector(; kws...) = Sector((; kws...))

data(s::Sector) = s.data

Base.keys(S::Sector) = keys(data(S))
Base.values(S::Sector) = values(data(S))
Base.getindex(S::Sector, args...) = getindex(data(S), args...)
Base.isempty(S::Sector) = isempty(data(S))

function Base.iterate(s::Sector, state=1)
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
function ⊗(A::Sector, B::Sector)
  qs = [A]
  #n=1 #DEBUG
  for (la, lb) in zip(intersect(A, B), intersect(B, A))
    #println()
    @assert la[1] == lb[1]
    fused_vals = ⊗(la[2], lb[2])
    #@show fused_vals
    function make_nt(s::Symbol,v) 
      NamedTuple{(s,)}((v,))
    end
    qs = [union(Sector(make_nt(la[1],v)),q) for v in fused_vals for q in qs]
    #println("Step $n qs = ",qs); n+=1
  end
  #println("Before last step, qs = ",qs)
  # Include sectors of B not in A
  qs = [union(q,B) for q in qs]
  #println("After last step, qs = ",qs)
  return qs
end

function Base.:(==)(A::Sector, B::Sector)
  common_labels = zip(intersect(A, B), intersect(B, A))
  common_labels_match = all(nl -> (nl[1] == nl[2]), common_labels)
  unique_labels_zero = all(l -> istrivial(l[2]), symdiff(A, B))
  return common_labels_match && unique_labels_zero
end

⊕(a::Sector, b::Sector) = [a, b]
⊕(v::Vector{<:Sector}, b::Sector) = vcat(v, b)
⊕(a::Sector, v::Vector{<:Sector}) = vcat(a, v)

function Base.show(io::IO, v::Vector{<:Sector})
  isempty(v) && return nothing
  symbol = ""
  for s in v
    print(io, symbol, s)
    symbol = " ⊕ "
  end
end

function Base.show(io::IO, s::Sector)
  if isempty(s)
    print(io, "Sector()")
  else
    print(io, "Sector", data(s))
  end
end
