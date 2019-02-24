
const LocalState = Union{Int,String}

struct InitState
  sts::Vector{LocalState}
  sites::SiteSet
  InitState(sites::SiteSet) = new(fill("",length(sites)),sites)
  InitState(sites::SiteSet,default::LocalState) = new(fill(default,length(sites)),sites)
  InitState(sites::SiteSet,def_vec::Vector{LocalState}) = new(def_vec, sites)
end

getindex(is::InitState,n::Integer) = getindex(is.sts,n)
setindex!(is::InitState,i::LocalState,n::Integer) = setindex!(is.sts,i,n)

site(is::InitState,n::Integer) = is.sites[n]
state(is::InitState,n::Integer)::IndexVal = state(site(is,n),sts[n])

length(is::InitState) = length(is.sts)

function show(io::IO,
              is::InitState)
  print(io,"InitState")
  (length(is) > 0) && print(io,"\n")
  for n=1:length(is)
    print(io,"  $(site(is,n)) ")
    if typeof(is[n])==Int
      print(io,"$(is[n])\n")
    else
      print(io,"\"$(is[n])\"\n")
    end
  end
end
