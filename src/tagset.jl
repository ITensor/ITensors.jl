
const Tag = SmallString

struct TagSet
  tags::Vector{String}
  TagSet() = new(Vector{String}())
  TagSet(tags::Vector{String}) = new(sort(tags))
end

function TagSet(tags::String)
  vectags = split(tags,",")
  filter!(s->s!="",vectags)
  return TagSet(String.(vectags))
end

length(T::TagSet) = length(T.tags)
getindex(T::TagSet,n::Int) = T.tags[n]

copy(ts::TagSet) = TagSet(ts.tags)

iterate(ts::TagSet,state::Int=1) = iterate(ts.tags,state)

convert(::Type{TagSet},x::String) = TagSet(x)
convert(::Type{TagSet},x::TagSet) = x

==(ts1::TagSet,ts2::TagSet) = (ts1.tags==ts2.tags)

in(tag::String, ts::TagSet) = in(tag, ts.tags)
function in(ts1::TagSet, ts2::TagSet)
  for t in ts1.tags
    !in(t,ts2.tags) && return false
  end
  return true
end

hastags(T::TagSet,ts::TagSet) = in(ts,T)
hastags(ts::TagSet,s::String) = in(TagSet(s),ts)

#TODO: optimize this code to not
#scan through all of the tags so many times
function addtags(ts::TagSet,tsadd::TagSet)
  res = copy(ts)
  #TODO: interface for iterating through tags
  for t in tsadd.tags
    t∉res.tags && push!(res.tags,t)
  end
  return TagSet(res.tags)
end

#TODO: optimize this function
function removetags(ts::TagSet,tsremove::TagSet)
  res = copy(ts)
  for t in tsremove.tags
    t∈res.tags && deleteat!(res.tags,findfirst(isequal(t),res.tags))
  end
  return res
end

#∈(tag::String, ts::TagSet) = in(tag, ts)

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
