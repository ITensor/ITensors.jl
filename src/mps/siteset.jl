export Site,
       ind,
       op,
       site,
       state,
       BasicSite,
       SiteSet,
       replaceBond!

abstract type Site end

state(s::Index,::Site,n::Int)::IndexVal = s(n)

defaultTags(::Site,n::Int) = TagSet("Site,n=$n")

dim(::Site) = throw(ErrorException("method dim not defined for abstract Site type"))


struct BasicSite <: Site end


const SiteSetStorage = Vector{Tuple{Index,Site}}

struct SiteSet
  store::SiteSetStorage

  SiteSet() = new(SiteSetStorage())

  SiteSet(N::Integer) = new(SiteSetStorage(undef,N))

  function SiteSet(N::Integer, d::Integer)
    store_ = SiteSetStorage(undef,N)
    for n=1:N
      ts = defaultTags(BasicSite(),n)
      store_[n] = (Index(d,ts),BasicSite())
    end
    new(store_)
  end
end

length(s::SiteSet) = length(s.store)
getindex(s::SiteSet,n::Integer)::Index = s.store[n][1]
siteType(s::SiteSet,n::Int) = s.store[n][2]
eachindex(s::SiteSet) = eachindex(s.store)

function setSite!(sset::SiteSet,n::Int,s::Site)
  i = Index(dim(s),defaultTags(s,n))
  sset.store[n] = (i,s)
end

function op(sset::SiteSet,
            opname::AbstractString,
            n::Int)::ITensor
  s = sset[n]
  sP = s'

  opname = strip(opname)

  Op = ITensor(dag(s),sP)
  if opname == "Id"
    for n=1:dim(s)
      Op[dag(s)(n),sP(n)] = 1.0
    end
  else
    # Interpret operator names joined by *
    # as acting on the same site
    starpos = findfirst("*",opname)
    if !isnothing(starpos)
      op1 = opname[1:starpos.start-1]
      op2 = opname[starpos.start+1:end]
      return multSiteOps(op(sset,op1,n),op(sset,op2,n))
    end
    return operator(s,siteType(sset,n),opname)
  end
  return Op
end

function show(io::IO,
              sset::SiteSet)
  print(io,"SiteSet")
  (length(sset) > 0) && print(io,"\n")
  for i=1:length(sset)
    println(io,"  $(sset[i]) $(typeof(siteType(sset,i)))")
  end
end

function state(sset::SiteSet,
               n::Integer,
               st::Union{Int,String})::IndexVal
  return state(sset[n],siteType(sset,n),st)
end
