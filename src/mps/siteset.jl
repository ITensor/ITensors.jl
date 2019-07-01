export Site,
       ind,
       op,
       BasicSite,
       SiteSet,
       replaceBond!

abstract type Site end

ind(st::Site) = st.s

function operator(s::Site,opname::String)::ITensor
  error("Operator name $opname not recognized for generic site")
  return ITensor()
end

function op(site::Site,
            opname::String)::ITensor
  s = ind(site)
  sP = s'

  Op = ITensor(dag(s),sP)
  if opname == "Id"
    for n=1:dim(s)
      Op[dag(s)(n),sP(n)] = 1.0
    end
  else
    return operator(site,opname)
  end
  return Op
end

struct BasicSite <: Site
  s::Index
  BasicSite(i::Index) = new(i)
end
BasicSite(d::Int) = BasicSite(Index(d,"Site"))
BasicSite(d::Int,n::Int) = BasicSite(Index(d,"Site,n=$n"))

struct SiteSet
  sites::Vector{Site}

  SiteSet() = new(Vector{Site}())

  SiteSet(N::Integer) = new(Vector{Site}(undef,N))

  function SiteSet(N::Integer, d::Integer)
    inds_ = Vector{Site}(undef,N)
    for n=1:N
      inds_[n] = BasicSite(d,n)
    end
    new(inds_)
  end
end

length(s::SiteSet) = length(s.sites)
getindex(s::SiteSet,n::Integer) = ind(s.sites[n])
op(s::SiteSet,opname::String,n::Int) = op(s.sites[n],opname)
set(s::SiteSet,n::Int,ns::Site) = (s.sites[n] = ns)

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
