
const Tag = SmallString

struct TagSet
  tags::Vector{String}
  plev::Int
  TagSet() = new(String[],-1)
  TagSet(tags::Vector{String},plev::Int=-1) = new(sort(tags),plev)
end

function TagSet(tags::AbstractString)
  vectags = split(tags,",")
  #If not specified, prime level starts at -1
  plev = -1
  plev_position = -1
  filter!(s->s!="",vectags)
  for i ∈ eachindex(vectags)
    plevtemp = tryparse(UInt,vectags[i])
    if !isnothing(plevtemp)
      plev_position≥0 && error("Cannot create a TagSet with more than two prime tags.")
      plev_position = i
      plev = Int(plevtemp)
    end
  end
  plev_position≥0 && deleteat!(vectags,plev_position)
  return TagSet(String.(vectags),plev)
end

length(T::TagSet) = length(T.tags)
getindex(T::TagSet,n::Int) = T.tags[n]
plev(T::TagSet) = T.plev
prime(T::TagSet,plinc::Int=1) = TagSet(copy(T.tags),plev(T)+plinc)
setprime(T::TagSet,pl::Int) = TagSet(copy(T.tags),pl::Int)

copy(ts::TagSet) = TagSet(copy(ts.tags),plev(ts))

iterate(ts::TagSet,state::Int=1) = iterate(ts.tags,state)

convert(::Type{TagSet},x::String) = TagSet(x)
convert(::Type{TagSet},x::TagSet) = x

==(ts1::TagSet,ts2::TagSet) = (ts1.tags==ts2.tags && plev(ts1)==plev(ts2))

function in(ts1::TagSet, ts2::TagSet)
  for t in ts1.tags
    !in(t,ts2.tags) && return false
  end
  (plev(ts1)≥0 && plev(ts1)≠plev(ts2)) && return false
  return true
end
in(s::String, ts::TagSet) = in(TagSet(s), ts)

hastags(T::TagSet,ts::TagSet) = in(ts,T)
hastags(ts::TagSet,s::String) = in(TagSet(s),ts)

#TODO: optimize this code to not
#scan through all of the tags so many times
function addtags(ts::TagSet,tsadd::TagSet,tsmatch::TagSet=TagSet())
  (plev(ts)≥0 && plev(tsadd)≥0) && error("In addtags(::TagSet,...), cannot add a prime level")
  restags = copy(ts.tags)
  resplev = ts.plev
  (tsmatch≠TagSet() && tsmatch∉ts) && return TagSet(restags,resplev)
  #TODO: interface for iterating through tags
  for t in tsadd.tags
    t∉restags && push!(restags,t)
  end
  (plev(ts)<0 && plev(tsadd)≥0) && (resplev = plev(tsadd))
  return TagSet(restags,resplev)
end

#TODO: optimize this function
function removetags(ts::TagSet,tsremove::TagSet,tsmatch::TagSet=TagSet())
  plev(tsremove)≥0 && error("In removetags(::TagSet,...), cannot remove a prime level")
  restags = copy(ts.tags)
  resplev = ts.plev
  (tsmatch≠TagSet() && tsmatch∉ts) && return TagSet(restags,resplev)
  for t in tsremove.tags
    t∈restags && deleteat!(restags,findfirst(isequal(t),restags))
  end
  #(plev(ts)≥0 && plev(ts)==plev(tsremove)) && (resplev = -1)
  return TagSet(restags,resplev)
end

#TODO: optimize this function
function replacetags(ts::TagSet,tsremove::TagSet,tsadd::TagSet,tsmatch::TagSet=TagSet())
  if (plev(tsremove)≥0 && plev(tsadd)<0) || (plev(tsremove)<0 && plev(tsadd)≥0)
    error("In replacetags(::TagSet,...), cannot remove or add a prime level")
  end
  restags = copy(ts.tags)
  resplev = ts.plev
  (tsmatch≠TagSet() && tsmatch∉ts) && return TagSet(restags,resplev)
  for t in tsremove.tags
    t ∉ ts.tags && return TagSet(restags,resplev)
  end
  for t in tsremove.tags
    deleteat!(restags,findfirst(isequal(t),restags))
  end
  for t in tsadd.tags
    push!(restags,t)
  end
  (plev(ts)≥0 && plev(ts)==plev(tsremove)) && (resplev = plev(tsadd))
  return TagSet(restags,resplev)
end

function primestring(ts::TagSet)
  pl = plev(ts)
  if pl<0 return " (warning: no prime level)"
  elseif pl==0 return ""
  elseif pl > 3 return "'$pl"
  else return "'"^pl
  end
end

function show(io::IO, T::TagSet)
  print(io,"\"")
  lT = length(T)
  if lT > 0
    print(io,T[1])
    for n=2:lT
      print(io,",$(T[n])")
    end
  end
  print(io,"\"")
end
