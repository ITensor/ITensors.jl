
@generated function sort_keys(nt::NamedTuple{N}) where {N}
  return :(NamedTuple{$(Tuple(sort(collect(N))))}(nt))
end

@generated function intersect_keys(nt1::NamedTuple{N1}, nt2::NamedTuple{N2}) where {N1,N2}
  return :(NamedTuple{$(Tuple(intersect(N1, N2)))}(merge(nt2, nt1)))
end

union_keys(ns1::NamedTuple, ns2::NamedTuple) = Base.merge(ns2, ns1)

setdiff_keys(ns1::NamedTuple, ns2::NamedTuple) = Base.structdiff(ns1, ns2)

function symdiff_keys(ns1::NamedTuple, ns2::NamedTuple)
  return merge(Base.structdiff(ns1, ns2), Base.structdiff(ns2, ns1))
end
