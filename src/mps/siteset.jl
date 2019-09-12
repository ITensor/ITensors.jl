export AbstractSite,
       defaultTags,
       dim,
       ind,
       op,
       setSite!,
       site,
       siteType,
       state,
       SiteSet,
       replaceBond!

abstract type AbstractSite end

function state(::Type{<:AbstractSite},s::Index,str::String)::IndexVal  
  throw(ErrorException("method state accepting String not defined for AbstractSite type"))
end

defaultTags(::Type{<:AbstractSite},n::Int) = TagSet("Site,n=$n")

dim(::Type{<:AbstractSite}) = throw(ErrorException("method dim not defined for AbstractSite type"))


const SiteSetStorage = Vector{Tuple{Index,Type{<:AbstractSite}}}

struct SiteSet
  store::SiteSetStorage

  SiteSet() = new(SiteSetStorage())

  SiteSet(N::Integer) = new(SiteSetStorage(undef,N))

end

length(s::SiteSet) = length(s.store)
getindex(s::SiteSet,n::Integer)::Index = s.store[n][1]
siteType(s::SiteSet,n::Int) = s.store[n][2]
eachindex(s::SiteSet) = eachindex(s.store)

function setSite!(sset::SiteSet,
                  n::Int,
                  st::Type{<:AbstractSite})
  i = Index(dim(st),defaultTags(st,n))
  sset.store[n] = (i,st)
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
    return op(siteType(sset,n),s,opname)
  end
  return Op
end

function show(io::IO,
              sset::SiteSet)
  print(io,"SiteSet")
  (length(sset) > 0) && print(io,"\n")
  for i=1:length(sset)
    println(io,"  $(sset[i]) $(siteType(sset,i))")
  end
end

function state(sset::SiteSet,
               j::Integer,
               str::String)::IndexVal
  sn = state(siteType(sset,j),str)
  return sset[j](sn)
end

function state(sset::SiteSet,
               j::Integer,
               st::Integer)::IndexVal
  return sset[j](st)
end
