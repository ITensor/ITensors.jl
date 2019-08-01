export Site,
       ind,
       op,
       site,
       state,
       BasicSite,
       SiteSet,
       replaceBond!

abstract type Site end

ind(st::Site) = st.s

state(site::Site,n::Integer) = ind(site)(n)

operator(s::Site,opname::String)::ITensor = throw(ArgumentError("Operator name $opname not recognized for generic site"))

function op(site::Site,
            opname::AbstractString)::ITensor
  s = ind(site)
  sP = s'

  opname = strip(opname)

  Op = ITensor(dag(s),sP)
  if opname == "Id"
    for n=1:dim(s)
      Op[dag(s)(n),sP(n)] = 1.0
    end
  else
    starpos = findfirst("*",opname)
    if !isnothing(starpos)
      op1 = opname[1:starpos.start-1]
      op2 = opname[starpos.start+1:end]
      return multSiteOps(op(site,op1),op(site,op2))
    end
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
site(s::SiteSet,n::Int) = s.sites[n]
eachindex(s::SiteSet) = eachindex(s.sites)

function show(io::IO,
              sites::SiteSet)
  print(io,"SiteSet")
  (length(sites) > 0) && print(io,"\n")
  for i=1:length(sites)
    println(io,"  $(sites[i])")
  end
end

function state(sset::SiteSet,
               n::Integer,
               st::Union{Int,String})::IndexVal
  return state(sset.sites[n],st)
end
