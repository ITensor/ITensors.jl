
abstract type SiteSet end

function show(io::IO,
              sites::SiteSet)
  print(io,"SiteSet")
  (length(sites) > 0) && print(io,"\n")
  for i=1:length(sites)
    println(io,"  $(sites[i])")
  end
end

length(sites::SiteSet) = length(inds(sites))
getindex(sites::SiteSet,i::Integer) = getindex(inds(sites),i)

function state(sites::SiteSet,
               n::Integer,
               st::Integer)::IndexVal
  return sites[n](st)
end

function state(sites::SiteSet,
               n::Integer,
               st::String)::IndexVal
  error("String version of 'state' SiteSet function not defined for this site set type")
  return sites[1](1)
end

struct Sites <: SiteSet
  inds::IndexSet
  Sites() = new(IndexSet())
  function Sites(N::Integer, d::Integer)
    inds_ = IndexSet(N)
    for i=1:N
      inds_[i] = Index(d,"$i,Site")
    end
    new(inds_)
  end
end

inds(s::Sites) = s.inds

