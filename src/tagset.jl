
const Tag = SmallString
const maxTagLength = smallLength
const MTagStorage = MSmallStringStorage # A mutable tag storage
const IntTag = IntSmallString  # An integer that can be cast to a Tag
const maxTags = 4
const TagSetStorage = SVector{maxTags,IntTag}
const MTagSetStorage = MVector{maxTags,IntTag}  # A mutable tag storage

struct TagSet
  data::TagSetStorage
  length::Int
  function TagSet() 
    ts = TagSetStorage(ntuple(_ -> IntTag(0), Val(4)))
    new(ts, 0)
  end
  TagSet(tags::TagSetStorage, len::Int) = new(tags, len)
end

TagSet(ts::TagSet) = ts

function TagSet(t::Tag)
  ts = MTagSetStorage(ntuple(_ -> IntTag(0), Val(maxTags)))
  ts[1] = IntSmallString(t)
  return TagSet(TagSetStorage(ts), 1)
end

macro ts_str(s)
  TagSet(s)
end

"""
    not(::TagSet)
    !(::TagSet)

Create a wrapper around a TagSet representing
the set of indices that do not contain that TagSet.
"""
not(ts::TagSet) = Not(ts)
Base.:!(ts::TagSet) = Not(ts)

not(ts::AbstractString) = Not(ts)

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

function _addtag!(ts::MTagSetStorage, ntags::Int, tag::IntTag)
  t = Tag(tag)
  if !isnull(t)
    if isint(t)
      error("Cannot use a bare integer as a tag.")
    else
      ntags = _addtag_ordered!(ts, ntags,tag)  
    end
  end
  return ntags
end

function reset!(v::MTagStorage, nchar::Int)
  for i = 1:nchar
    @inbounds v[i] = IntChar(0)
  end
end

function TagSet(str::AbstractString)
  # Mutable fixed-size vector as temporary Tag storage
  current_tag = MTagStorage(ntuple(_ -> IntChar(0),Val(maxTagLength)))
  # Mutable fixed-size vector as temporary TagSet storage
  ts = MTagSetStorage(ntuple(_ -> IntTag(0),Val(maxTags)))
  nchar = 0
  ntags = 0
  for current_char in str
    if current_char == ','
      if nchar != 0
        ntags = _addtag!(ts,ntags,cast_to_uint(current_tag))
        # Reset the current tag
        reset!(current_tag,nchar)
        nchar = 0
      end
    elseif current_char != ' ' # TagSet constructor ignores whitespace
      nchar == maxTagLength && error("Currently, tags can only have up to $maxTagLength characters")
      nchar += 1
      @inbounds current_tag[nchar] = current_char
    end
  end
  # Store the final tag
  if nchar != 0
    ntags = _addtag!(ts,ntags,cast_to_uint(current_tag))
  end
  return TagSet(TagSetStorage(ts),ntags)
end

Base.convert(::Type{TagSet}, str::String) = TagSet(str)

"""
ITensors.data(T::TagSet)

Get the raw storage of the TagSet.

This is an internal function, please inform the
developers of ITensors.jl if there is functionality
you would like for TagSet that is not currently
available.
"""
data(T::TagSet) = T.data

Base.length(T::TagSet) = T.length
Base.getindex(T::TagSet,n::Int) = Tag(getindex(data(T),n))
Base.copy(ts::TagSet) = TagSet(data(ts),length(ts))

function Base.:(==)(ts1::TagSet,ts2::TagSet)
  l1 = length(ts1)
  l2 = length(ts2)
  l1 != l2 && return false
  for n in 1:l1
    @inbounds ts1[n] != ts2[n] && return false
  end
  return true
end

function hastag(ts::TagSet, tag)
  l = length(ts)
  l < 1 && return false
  for n = 1:l
    @inbounds Tag(tag) == ts[n] && return true
  end
  return false
end

function hastags(ts2::TagSet, tags1)
  ts1 = TagSet(tags1)
  l1 = length(ts1)
  l2 = length(ts2)
  l1 > l2 && return false
  for n1 = 1:l1
    @inbounds !hastag(ts2,ts1[n1]) && return false
  end
  return true
end

function addtags(ts::TagSet, tagsadd)
  if length(ts) == maxTags
    throw(ErrorException("Cannot add tag: TagSet already maximum size"))
  end
  tsadd = TagSet(tagsadd)
  res_ts = MVector(data(ts))
  ntags = length(ts)
  for n = 1:length(tsadd)
    @inbounds ntags = _addtag_ordered!(res_ts, ntags,IntSmallString(tsadd[n]))
  end
  return TagSet(TagSetStorage(res_ts),ntags)
end

function _removetag!(ts::MTagSetStorage, ntags::Int, t::Tag)
  for n = 1:ntags
    if @inbounds Tag(ts[n]) == t
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
  res_ts = MVector(data(ts))
  ntags = length(ts)
  for n=1:length(tsremove)
    @inbounds ntags = _removetag!(res_ts, ntags, tsremove[n])
  end
  return TagSet(TagSetStorage(res_ts),ntags)
end

#TODO: optimize this function
function replacetags(ts::TagSet, tagsremove, tagsadd)
  tsremove = TagSet(tagsremove)
  tsadd = TagSet(tagsadd)
  res_ts = MVector(data(ts))
  ntags = length(ts)
  # The TagSet must have the tags to be replaced
  !hastags(ts,tsremove) && return ts
  for n = 1:length(tsremove)
    @inbounds ntags = _removetag!(res_ts, ntags, tsremove[n])
  end
  for n = 1:length(tsadd)
    @inbounds ntags = _addtag_ordered!(res_ts, ntags,IntSmallString(tsadd[n]))
  end
  return TagSet(TagSetStorage(res_ts),ntags)
end

function tagstring(T::TagSet)
  res = ""
  N = length(T)
  N == 0 && return res
  for n=1:N-1
    res *= "$(Tag(T[n])),"
  end
  res *= "$(Tag(T[N]))"
  return res
end

"""
    iterate(is::TagSet[, state])

Iterate over the Tag's in a TagSet.

# Example
```jldoctest
julia> using ITensors;

julia> tagset = TagSet("l, tags");

julia> for tag in tagset
         println(tag)
       end
l
tags
```
"""
Base.iterate(ts::TagSet, state) = state < length(ts) ? (ts[state + 1], state + 1) : nothing

Base.iterate(ts::TagSet) = (ts[1], 1)

commontags(ts::TagSet) = ts

function commontags(ts1::TagSet, ts2::TagSet)
  ts3 = TagSet()
  N1 = length(ts1)
  for n1 in 1:N1
    t1 = ts1[n1]
    if hastag(ts2, t1)
      ts3 = addtags(ts3, t1)
    end
  end
  return ts3
end

function commontags(ts1::TagSet, ts2::TagSet,
                    ts3::TagSet, ts::TagSet...)
  return commontags(commontags(ts1, ts2), ts3, ts...)
end

function Base.show(io::IO, T::TagSet)
  print(io, "\"$(tagstring(T))\"")
end

function readcpp(io::IO,::Type{TagSet}; kwargs...)
  format = get(kwargs,:format,"v3")
  ts = TagSet()
  if format=="v3"
    mstore = MTagSetStorage(ntuple(_ -> IntTag(0),Val(maxTags)))
    ntags = 0
    for n=1:4
      t = readcpp(io,Tag;kwargs...)
      if t != Tag()
        ntags = _addtag_ordered!(mstore,ntags,IntSmallString(t))
      end
    end
    ts = TagSet(TagSetStorage(mstore),ntags)
  else
    throw(ArgumentError("read TagSet: format=$format not supported"))
  end
  return ts
end

function HDF5.write(parent::Union{HDF5File,HDF5Group},
                    name::AbstractString,
                    T::TagSet)
  g = g_create(parent,name)
  attrs(g)["type"] = "TagSet"
  attrs(g)["version"] = 1
  write(g,"tags", tagstring(T))
end

function HDF5.read(parent::Union{HDF5File,HDF5Group},
                   name::AbstractString,
                   ::Type{TagSet})
  g = g_open(parent,name)
  if read(attrs(g)["type"]) != "TagSet"
    error("HDF5 group '$name' does not contain TagSet data")
  end
  tstring = read(g,"tags")
  return TagSet(tstring)
end

