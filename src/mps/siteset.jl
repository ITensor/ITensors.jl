
struct SiteSet
  inds::IndexSet
  SiteSet() = new(IndexSet())
  function SiteSet(N::Integer, d::Integer)
    inds_ = IndexSet(N)
    for i=1:N
      inds_[i] = Index(d)
    end
    new(inds_)
  end
end

length(sites::SiteSet) = length(sites.inds)
getindex(sites::SiteSet,i::Integer) = getindex(sites.inds,i)

import Base.show
function show(io::IO,
              sites::SiteSet)
  println(io,"SiteSet")
  for i=1:length(sites)
    println(io,"$i  $(sites[i])")
  end
end

