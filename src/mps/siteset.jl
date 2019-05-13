
struct SiteSet
  inds::IndexSet
  SiteSet() = new(IndexSet())
  function SiteSet(N::Integer, d::Integer)
    inds_ = IndexSet(N)
    for n=1:N
      inds_[n] = Index(d,"n=$n,Site")
    end
    new(inds_)
  end
end

inds(s::SiteSet) = s.inds
length(s::SiteSet) = length(inds(s))
getindex(s::SiteSet,n::Integer) = getindex(inds(s),n)

function show(io::IO,
              sites::SiteSet)
  print(io,"SiteSet")
  (length(sites) > 0) && print(io,"\n")
  for i=1:length(sites)
    println(io,"  $(sites[i])")
  end
end

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


abstract type Site end

