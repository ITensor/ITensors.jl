
@generated function nt_sort(nt::NamedTuple{N}) where {N}
  return :(NamedTuple{$(Tuple(sort(collect(N))))}(nt))
end

@generated function nt_intersect(nt1::NamedTuple{N1}, nt2::NamedTuple{N2}) where {N1,N2}
  return :(NamedTuple{$(Tuple(intersect(N1, N2)))}(merge(nt2, nt1)))
end

nt_union(ns1::NamedTuple, ns2::NamedTuple) = Base.merge(ns2, ns1)

nt_setdiff(ns1::NamedTuple, ns2::NamedTuple) = Base.structdiff(ns1, ns2)

function nt_symdiff(ns1::NamedTuple, ns2::NamedTuple)
  return merge(Base.structdiff(ns1, ns2), Base.structdiff(ns2, ns1))
end
