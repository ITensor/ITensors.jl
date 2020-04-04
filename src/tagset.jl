export TagSet,
       addtags,
       hastags,
       Tag

const Tag = SmallString
const maxTagLength = smallLength
const MTagStorage = MSmallStringStorage # A mutable tag storage
const IntTag = IntSmallString  # An integer that can be cast to a Tag
const maxTags = 4
const TagSetStorage = SVector{maxTags,IntTag}
const MTagSetStorage = MVector{maxTags,IntTag}  # A mutable tag storage

struct TagSet
  tags::TagSetStorage
  length::Int
  function TagSet() 
    ts = TagSetStorage(ntuple(_ -> IntTag(0),Val(4)))
    new(ts,0)
  end
  TagSet(tags::TagSetStorage,len::Int) = new(tags,len)
end

TagSet(ts::TagSet) = ts

not(ts::Union{AbstractString,TagSet}) = Not(TagSet(ts))

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
  for n = 1:length(str)
    @inbounds current_char = str[n]
    if current_char == ','
      if nchar != 0
        ntags = _addtag!(ts,ntags,cast_to_uint64(current_tag))
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
    ntags = _addtag!(ts,ntags,cast_to_uint64(current_tag))
  end
  return TagSet(TagSetStorage(ts),ntags)
end

Base.convert(::Type{TagSet}, str::String) = TagSet(str)

Tensors.store(T::TagSet) = T.tags
Base.length(T::TagSet) = T.length
Base.getindex(T::TagSet,n::Int) = Tag(getindex(store(T),n))
Base.copy(ts::TagSet) = TagSet(store(ts),length(ts))

# Cast SVector of IntTag of length 4 to SVector of UInt128 of length 2
# This is to make TagSet comparison a little bit faster
function cast_to_uint128(a::TagSetStorage)
  return unsafe_load(convert(Ptr{SVector{2,UInt128}},pointer_from_objref(MTagSetStorage(a))))
end

function Base.:(==)(ts1::TagSet,ts2::TagSet)
  # Block the bits together to make the comparison faster
  return cast_to_uint128(store(ts1)) == cast_to_uint128(store(ts2))
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
  res_ts = MVector(store(ts))
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
  res_ts = MVector(store(ts))
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
  res_ts = MVector(store(ts))
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
    print(io,T[1])
    for n=2:lT
      print(io,",$(T[n])")
    end
  end
  print(io,")")
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
  write(g,"tags",tagstring(T))
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

@deprecate tags(t::TagSet) store(t::TagSet)

