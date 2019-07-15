export TagSet

const Tag = SmallString
const MTagStorage = MSmallStringStorage # A mutable tag storage
const IntTag = IntSmallString  # An integer that can be cast to a Tag
const TagSetStorage = SVector{4,IntTag}
const MTagSetStorage = MVector{4,IntTag}  # A mutable tag storage

struct TagSet
  tags::TagSetStorage
  plev::Int
  length::Int
  function TagSet() 
    ts = TagSetStorage(ntuple(_ -> IntTag(0),Val(4)))
    plev = -1
    new(ts,plev,0)
  end
  TagSet(tags::TagSetStorage,plev::Int,len::Int) = new(tags,plev,len)
end

function TagSet(tags::TagSetStorage,plev::Int=-1)
  len = 0
  while tags[len+1] ≠ IntTag(0)
    len += 1
  end
  TagSet(tags,plev,len)
end

TagSet(ts::TagSet) = ts

function _hastag(ts::MTagSetStorage, ntags::Int, tag::IntTag)
  for n = 1:ntags
    @inbounds ts[n] == tag && return true
  end
  return false
end

function _addtag_ordered!(ts::MTagSetStorage, ntags::Int, tag::IntTag)
  if ntags == 0 || tag > @inbounds ts[ntags]
    @inbounds setindex!(ts,tag,ntags+1)
  else
    # check for repeated tags
    _hastag(ts,ntags,tag) && return ntags
    pos = ntags+1   # position new tag should go
    while pos > 1 && tag < @inbounds ts[pos-1]
      pos -= 1
      @inbounds setindex!(ts,ts[pos],pos+1)
    end
    @inbounds setindex!(ts,tag,pos)
  end
  return ntags+1
end

function _addtag!(ts::MTagSetStorage, plev::Int, ntags::Int, tag::IntTag)
  plnew = plev
  t = Tag(tag)
  if length(t) > 0
    if isint(t)
      plev ≠ -1 && error("You can only make a TagSet with one prime level/integer tag.")
      plnew = parse(Int,t)
    else
      ntags = _addtag_ordered!(ts, ntags, tag)  
    end
  end
  return plnew, ntags
end

isnull(v::MTagStorage) = v[0] == IntChar(0)

function reset!(v::MTagStorage, nchar::Int)
  for i = 1:nchar
    @inbounds v[i] = IntChar(0)
  end
end

function TagSet(str::AbstractString)
  # Mutable fixed-size vector as temporary Tag storage
  current_tag = MTagStorage(ntuple(_ -> IntChar(0),Val(8)))
  # Mutable fixed-size vector as temporary TagSet storage
  ts = MTagSetStorage(ntuple(_ -> IntTag(0),Val(4)))
  nchar = 0
  ntags = 0
  plev = -1
  for n = 1:length(str)
    @inbounds current_char = str[n]
    if current_char == ','
      if nchar ≠ 0
        plev, ntags = _addtag!(ts,plev,ntags,cast_to_uint64(current_tag))
        # Reset the current tag
        reset!(current_tag,nchar)
        nchar = 0
      end
    elseif current_char ≠ ' '
      nchar += 1
      @inbounds current_tag[nchar] = current_char
    end
  end
  # Store the final tag
  if nchar ≠ 0
    plev, ntags = _addtag!(ts,plev,ntags,cast_to_uint64(current_tag))
  end
  return TagSet(TagSetStorage(ts),plev,ntags)
end

convert(::Type{TagSet}, str::String) = TagSet(str)

tags(T::TagSet) = T.tags
plev(T::TagSet) = T.plev
Base.length(T::TagSet) = T.length
Base.getindex(T::TagSet,n::Int) = getindex(tags(T),n)
Base.setindex(T::TagSet,val,n::Int) = TagSet(setindex(tags(T),val,n),plev(T),length(T))
Base.copy(ts::TagSet) = TagSet(tags(ts),plev(ts),length(ts))

setprime(ts::TagSet,pl::Int) = TagSet(tags(ts),pl,length(ts))
function prime(T::TagSet,plinc::Int=1)
  if plev(T) < 0
    return setprime(T,plinc)
  else
    return setprime(T,plev(T)+plinc)
  end
end

# TODO: define iteration?
#Base.iterate(ts::TagSet,state::Int=1) = iterate(tags(ts),state)

# TODO: define `in` in terms of hastags?
#Base.in(s, ts::TagSet) = hastags(ts, TagSet(s))

# Cast SVector of IntTag of length 4 to SVector of UInt128 of length 2
# This is to make TagSet comparison a little bit faster
function cast_to_uint128(a::TagSetStorage)
  return unsafe_load(convert(Ptr{SVector{2,UInt128}},pointer_from_objref(MTagSetStorage(a))))
end

import Base.==
function ==(ts1::TagSet,ts2::TagSet)
  plev(ts1) ≠ plev(ts2) && return false
  # Block the bits together to make the comparison faster
  return cast_to_uint128(tags(ts1)) == cast_to_uint128(tags(ts2))
end

function hastag(ts::TagSet, t::IntTag)
  l = length(ts)
  l < 1 && return false
  for n = 1:l
    @inbounds t == ts[n] && return true
  end
  return false
end

function hastags(ts2::TagSet, tags1)
  ts1 = TagSet(tags1)
  (plev(ts1) ≥ 0 && plev(ts1) ≠ plev(ts2)) && return false
  l1 = length(ts1)
  l2 = length(ts2)
  l1 > l2 && return false
  for n1 = 1:l1
    @inbounds !hastag(ts2,ts1[n1]) && return false
  end
  return true
end

function addtags(ts::TagSet, tagsadd)
  tsadd = TagSet(tagsadd)
  ( plev(ts) ≥ 0 && plev(tsadd) ≥ 0 ) && error("In addtags(::TagSet,...), cannot add a prime level")
  res_ts = MVector(tags(ts))
  res_plev = plev(ts)
  ntags = length(ts)
  for n = 1:length(tsadd)
    @inbounds ntags = _addtag_ordered!(res_ts, ntags, tsadd[n])
  end
  (plev(ts) < 0 && plev(tsadd)≥0) && (res_plev = plev(tsadd))
  return TagSet(TagSetStorage(res_ts),res_plev,ntags)
end

function _removetag!(ts::MTagSetStorage, ntags::Int, t::IntTag)
  for n = 1:ntags
    if @inbounds ts[n] == t
      for j = n:ntags-1
        @inbounds ts[j] = ts[j+1]
      end
      @inbounds ts[ntags] = IntTag(0)
      return ntags -= 1
    end
  end
  return ntags
end

#TODO: optimize this function
function removetags(ts::TagSet, tagsremove)
  tsremove = TagSet(tagsremove)
  plev(tsremove) ≥ 0 && error("In removetags(::TagSet,...), cannot remove a prime level")
  res_ts = MVector(tags(ts))
  res_plev = plev(ts)
  ntags = length(ts)
  for n = 1:length(tsremove)
    @inbounds ntags = _removetag!(res_ts, ntags, tsremove[n])
  end
  return TagSet(TagSetStorage(res_ts),res_plev,ntags)
end

#TODO: optimize this function
function replacetags(ts::TagSet, tagsremove, tagsadd)
  tsremove = TagSet(tagsremove)
  tsadd = TagSet(tagsadd)
  if (plev(tsremove) ≥ 0 && plev(tsadd) < 0) || 
     (plev(tsremove) < 0 && plev(tsadd) ≥ 0)
    error("In replacetags(::TagSet,...), cannot remove or add a prime level")
  end
  res_ts = MVector(tags(ts))
  res_plev = plev(ts)
  ntags = length(ts)
  # The TagSet must have the same prime level as the one being replaced
  (res_plev ≠ plev(tsremove) && plev(tsremove) ≥ 0) && return ts
  # The TagSet must have the tags to be replaced
  !hastags(ts,tsremove) && return ts
  for n = 1:length(tsremove)
    @inbounds ntags = _removetag!(res_ts, ntags, tsremove[n])
  end
  for n = 1:length(tsadd)
    @inbounds ntags = _addtag_ordered!(res_ts, ntags, tsadd[n])
  end
  (plev(ts)≥0 && plev(ts)==plev(tsremove)) && (res_plev = plev(tsadd))
  return TagSet(TagSetStorage(res_ts),res_plev,ntags)
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
  length(T) == 0 && return res
  for n=1:length(T)-1
    res *= "$(Tag(T[n])),"
  end
  res *= "$(Tag(T[length(T)]))"
  return res
end

function show(io::IO, T::TagSet)
  print(io,"(")
  lT = length(T)
  if lT > 0
    print(io,Tag(T[1]))
    for n=2:lT
      print(io,",$(Tag(T[n]))")
    end
  end
  print(io,")")
  print(io,primestring(T))
end

export addtags,
       hastags,
       Tag
