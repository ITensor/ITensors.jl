
@generated function nt_sort(nt::NamedTuple{N}) where {N}
  return :(NamedTuple{$(Tuple(sort(collect(N))))}(nt))
end

@generated function nt_intersect(nt1::NamedTuple{N1}, nt2::NamedTuple{N2}) where {N1,N2}
  return :(NamedTuple{$(Tuple(intersect(N1,N2)))}(merge(nt2, nt1)))
end

abstract type AbstractNamedSet end

Base.union(ns1::AbstractNamedSet, ns2::AbstractNamedSet) = typeof(ns1)(merge(data(ns2), data(ns1)))

Base.intersect(ns1::AbstractNamedSet, ns2::AbstractNamedSet) = typeof(ns1)(nt_intersect(data(ns1), data(ns2)))

Base.setdiff(ns1::AbstractNamedSet, ns2::AbstractNamedSet) = typeof(ns1)(Base.structdiff(data(ns1), data(ns2)))

function Base.symdiff(ns1::AbstractNamedSet, ns2::AbstractNamedSet) 
  ndata = merge(Base.structdiff(data(ns1), data(ns2)), Base.structdiff(data(ns2), data(ns1)))
  return typeof(ns1)(ndata)
end

#Base.iterate(ns::AbstractNamedSet,args...) = Base.iterate(data(ns),args...)

Base.length(ns::AbstractNamedSet) = length(data(ns))
