
const Tag = SmallString

struct TagSet
  tags::Vector{String}
  plev::Int
  TagSet() = new(String[],-1)
  TagSet(tags::Vector{String},plev::Int=-1) = new(sort(tags),plev)
end

TagSet(ts::TagSet) = ts

function TagSet(tags::AbstractString)
  vectags = split(tags,",")
  #Remove all whitespace when creating a tag
  vectags = filter.(x ->!isspace(x),vectags)
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

tags(T::TagSet) = T.tags
plev(T::TagSet) = T.plev
length(T::TagSet) = length(tags(T))
getindex(T::TagSet,n::Int) = tags(T)[n]
copy(ts::TagSet) = TagSet(copy(tags(ts)),plev(ts))

setprime(T::TagSet,pl::Int) = TagSet(copy(tags(T)),pl)
prime(T::TagSet,plinc::Int) = setprime(T,plev(T)+plinc)

iterate(ts::TagSet,state::Int=1) = iterate(tags(ts),state)

==(ts1::TagSet,ts2::TagSet) = (tags(ts1) == tags(ts2) && 
                               plev(ts1) == plev(ts2))

function in(ts1::TagSet, ts2::TagSet)
  for t in tags(ts1)
    t ∉ tags(ts2) && return false
  end
  (plev(ts1) ≥ 0 && plev(ts1) ≠ plev(ts2)) && return false
  return true
end

in(s::String, ts::TagSet) = in(TagSet(s), ts)

hastags(T::TagSet,ts::TagSet) = in(ts,T)
hastags(ts::TagSet,s::String) = in(TagSet(s),ts)

#TODO: optimize this code to not
#scan through all of the tags so many times
function addtags(ts::TagSet, tagsadd)
  tsadd = TagSet(tagsadd)
  ( plev(ts) ≥ 0 && plev(tsadd) ≥ 0 ) && error("In addtags(::TagSet,...), cannot add a prime level")
  restags = copy(tags(ts))
  resplev = ts.plev
  #TODO: interface for iterating through tags
  for t ∈ tags(tsadd)
    t ∉ restags && push!(restags,t)
  end
  (plev(ts) < 0 && plev(tsadd)≥0) && (resplev = plev(tsadd))
  return TagSet(restags,resplev)
end

#TODO: optimize this function
function removetags(ts::TagSet, tagsremove)
  tsremove = TagSet(tagsremove)
  plev(tsremove) ≥ 0 && error("In removetags(::TagSet,...), cannot remove a prime level")
  restags = copy(tags(ts))
  resplev = ts.plev
  for t in tags(tsremove)
    t∈restags && deleteat!(restags,findfirst(isequal(t),restags))
  end
  #(plev(ts)≥0 && plev(ts)==plev(tsremove)) && (resplev = -1)
  return TagSet(restags,resplev)
end

#TODO: optimize this function
function replacetags(ts::TagSet, tagsremove, tagsadd)
  tsremove = TagSet(tagsremove)
  tsadd = TagSet(tagsadd)
  if (plev(tsremove) ≥ 0 && plev(tsadd) < 0) || 
     (plev(tsremove) < 0 && plev(tsadd) ≥ 0)
    error("In replacetags(::TagSet,...), cannot remove or add a prime level")
  end
  restags = copy(tags(ts))
  resplev = plev(ts)
  (resplev ≠ plev(tsremove) && plev(tsremove) ≥ 0) && return TagSet(restags,resplev)
  for t in tags(tsremove)
    t ∉ tags(ts) && return TagSet(restags,resplev)
  end
  for t in tags(tsremove)
    deleteat!(restags,findfirst(isequal(t),restags))
  end
  for t in tags(tsadd)
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

function tagstring(T::TagSet)
  res = ""
  length(T)==0 && return res
  for n=1:length(T)-1
    res *= "$(T[n]),"
  end
  res *= T[length(T)]
  return res
end

function show(io::IO, T::TagSet)
  print(io,"(")
  lT = length(T)
  if lT > 0
    print(io,T[1])
    for n=2:lT
      print(io,",$(T[n])")
    end
  end
  print(io,")")
  print(io,primestring(T))
end

export addtags,
       hastags,
       Tag
