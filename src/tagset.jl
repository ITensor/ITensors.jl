using BitIntegers

const IntTag = UInt256  # An integer that can be cast to a Tag
const MTagStorage = MVector{16,IntTag} # A mutable tag storage, holding 16 characters
const TagSetStorage{T,N} = SVector{N,T}
const MTagSetStorage{T,N} = MVector{N,T}  # A mutable tag storage

emptytag(::Type{IntTag}) = IntTag(0)
function empty_storage(::Type{TagSetStorage{T,N}}) where {T,N}
  return TagSetStorage(ntuple(_ -> emptytag(T), Val(N)))
end
function empty_storage(::Type{MTagSetStorage{T,N}}) where {T,N}
  return MTagSetStorage(ntuple(_ -> emptytag(T), Val(N)))
end

#TODO: decide which functions on TagSet should be made generic.
struct GenericTagSet{T,N}
  data::TagSetStorage{T,N}
  length::Int
  GenericTagSet{T,N}() where {T,N} = new(empty_storage(TagSetStorage{T,N}), 0)
  GenericTagSet{T,N}(tags::TagSetStorage{T,N}, len::Int) where {T,N} = new(tags, len)
end

GenericTagSet{T,N}(ts::GenericTagSet{T,N}) where {T,N} = ts

function GenericTagSet{T,N}(t::T) where {T,N}
  ts = empty_storage(MTagSetStorage{T,N})
  ts[1] = T(t)
  return GenericTagSet{T,N}(TagSetStorage(ts), 1)
end

#GenericTagSet{IntTag,N}(t::Tag) where {N} = GenericTagSet{IntTag,N}(IntTag(t))

function _hastag(ts::MTagSetStorage, ntags::Int, tag::IntTag)
  for n in 1:ntags
    @inbounds ts[n] == tag && return true
  end
  return false
end

function _addtag_ordered!(ts::MTagSetStorage, ntags::Int, tag::IntTag)
  if iszero(ntags) || tag > @inbounds ts[ntags]
    @inbounds setindex!(ts, tag, ntags + 1)
  else
    # check for repeated tags
    _hastag(ts, ntags, tag) && return ntags
    pos = ntags + 1   # position new tag should go
    while pos > 1 && tag < @inbounds ts[pos - 1]
      pos -= 1
      @inbounds setindex!(ts, ts[pos], pos + 1)
    end
    @inbounds setindex!(ts, tag, pos)
  end
  return ntags + 1
end

function _addtag!(ts::MTagSetStorage, ntags::Int, tag::IntTag)
  t = Tag(tag)
  # TODO: change to isempty, remove isnull
  if !isnull(t)
    ntags = _addtag_ordered!(ts, ntags, tag)
  end
  return ntags
end

function reset!(v::MTagStorage, nchar::Int)
  for i in 1:nchar
    @inbounds v[i] = IntTag(0)
  end
end

function strict_tags_error(str, maxlength, nchar)
  return error(
    "You are trying to make a TagSet from the String \"$(str)\". This has more than the maximum number of allowed tags ($maxlength), or has a tag that is longer than the longest allowed tag ($nchar). Either specify fewer or shorter tags, or use `ITensors.set_strict_tags!(false)` to disable this error, in which case extra tags or tag characters will be ignored.",
  )
end

function strict_tags_add_error(ts, tsadd, maxlength)
  return error(
    "You are trying to add the TagSet $tsadd to the TagSet $ts. The result would have more than the maximum number of allowed tags ($maxlength). Either specify fewer tags, or use `ITensors.set_strict_tags!(false)` to disable this error, in which case extra tags will be ignored.",
  )
end

function strict_tags_replace_error(ts, tsremove, tsadd, maxlength)
  return error(
    "You are trying to replace the TagSet $tsremove with the TagSet $tsadd in the TagSet $ts. The result would have more than the maximum number of allowed tags ($maxlength). Either specify fewer tags, or use `ITensors.set_strict_tags!(false)` to disable this error, in which case extra tags will be ignored.",
  )
end

function GenericTagSet{T,N}(str::AbstractString) where {T,N}
  # Mutable fixed-size vector as temporary Tag storage
  # TODO: refactor the Val here.
  current_tag = empty_storage(MTagStorage)
  # Mutable fixed-size vector as temporary TagSet storage
  ts = empty_storage(MTagSetStorage{T,N})
  nchar = 0
  ntags = 0
  for current_char in str
    if current_char == ','
      if nchar != 0
        if ntags < N
          ntags = _addtag!(ts, ntags, cast_to_uint(current_tag))
        elseif using_strict_tags()
          strict_tags_error(str, N, length(current_tag))
        end # else do nothing
        # Reset the current tag
        reset!(current_tag, nchar)
        nchar = 0
      end
    elseif current_char != ' ' # TagSet constructor ignores whitespace
      if nchar ≥ length(current_tag)
        if using_strict_tags()
          strict_tags_error(str, N, length(current_tag))
        else
          continue
        end
      end
      nchar += 1
      @inbounds current_tag[nchar] = current_char
    end
  end
  # Store the final tag
  if nchar != 0
    if ntags < N
      ntags = _addtag!(ts, ntags, cast_to_uint(current_tag))
    elseif using_strict_tags()
      strict_tags_error(str, N, length(current_tag))
    end # else do nothing
  end
  if ntags > N
    if using_strict_tags()
      strict_tags_error(str, N, length(current_tag))
    else
      ntags = N
    end
  end
  return GenericTagSet{T,N}(TagSetStorage(ts), ntags)
end

const TagSet = GenericTagSet{IntTag,4}

maxlength(::GenericTagSet{<:Any,N}) where {N} = N

macro ts_str(s)
  return TagSet(s)
end

Base.convert(::Type{TagSet}, str::String) = TagSet(str)

"""
    not(::TagSet)
    !(::TagSet)

Create a wrapper around a TagSet representing
the set of indices that do not contain that TagSet.
"""
not(ts::TagSet) = Not(ts)
Base.:!(ts::TagSet) = Not(ts)

not(ts::AbstractString) = Not(ts)

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
@propagate_inbounds getindex(T::TagSet, n::Integer) = Tag(data(T)[n])
Base.copy(ts::TagSet) = TagSet(data(ts), length(ts))

function Base.:(==)(ts1::TagSet, ts2::TagSet)
  l1 = length(ts1)
  l2 = length(ts2)
  l1 != l2 && return false
  for n in 1:l1
    @inbounds data(ts1)[n] != data(ts2)[n] && return false
  end
  return true
end

# Assumes it is an integer
function hastag(ts::TagSet, tag)
  l = length(ts)
  l < 1 && return false
  for n in 1:l
    @inbounds tag == data(ts)[n] && return true
  end
  return false
end

function hastags(ts2::TagSet, tags1)
  ts1 = TagSet(tags1)
  l1 = length(ts1)
  l2 = length(ts2)
  l1 > l2 && return false
  for n1 in 1:l1
    @inbounds !hastag(ts2, data(ts1)[n1]) && return false
  end
  return true
end

function addtags(ts::TagSet, tagsadd)
  tsadd = TagSet(tagsadd)
  if length(ts) == maxlength(ts)
    if hastags(ts, tsadd)
      return ts
    end
    if using_strict_tags()
      strict_tags_add_error(ts, tsadd, maxlength(ts))
    end
  end
  res_ts = MVector(data(ts))
  ntags = length(ts)
  for n in 1:length(tsadd)
    if ntags < maxlength(ts)
      @inbounds ntags = _addtag_ordered!(res_ts, ntags, data(tsadd)[n])
    elseif using_strict_tags()
      strict_tags_add_error(ts, tsadd, maxlength(ts))
    end
  end
  return TagSet(TagSetStorage(res_ts), ntags)
end

function _removetag!(ts::MTagSetStorage, ntags::Int, t)
  for n in 1:ntags
    if @inbounds ts[n] == t
      for j in n:(ntags - 1)
        @inbounds ts[j] = ts[j + 1]
      end
      @inbounds ts[ntags] = emptytag(IntTag)
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
  for n in 1:length(tsremove)
    @inbounds ntags = _removetag!(res_ts, ntags, data(tsremove)[n])
  end
  return TagSet(TagSetStorage(res_ts), ntags)
end

#TODO: optimize this function
function replacetags(ts::TagSet, tagsremove, tagsadd)
  tsremove = TagSet(tagsremove)
  tsadd = TagSet(tagsadd)
  res_ts = MVector(data(ts))
  ntags = length(ts)
  # The TagSet must have the tags to be replaced
  !hastags(ts, tsremove) && return ts
  for n in 1:length(tsremove)
    @inbounds ntags = _removetag!(res_ts, ntags, data(tsremove)[n])
  end
  for n in 1:length(tsadd)
    if ntags < maxlength(ts)
      @inbounds ntags = _addtag_ordered!(res_ts, ntags, data(tsadd)[n])
    elseif using_strict_tags()
      strict_tags_replace_error(ts, tsremove, tsadd, maxlength(ts))
    end
  end
  return TagSet(TagSetStorage(res_ts), ntags)
end

function tagstring(T::TagSet)
  res = ""
  N = length(T)
  N == 0 && return res
  for n in 1:(N - 1)
    res *= "$(Tag(data(T)[n])),"
  end
  res *= "$(Tag(data(T)[N]))"
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
    t1 = data(ts1)[n1]
    if hastag(ts2, t1)
      ts3 = addtags(ts3, t1)
    end
  end
  return ts3
end

function commontags(ts1::TagSet, ts2::TagSet, ts3::TagSet, ts::TagSet...)
  return commontags(commontags(ts1, ts2), ts3, ts...)
end

function Base.show(io::IO, T::TagSet)
  return print(io, "\"$(tagstring(T))\"")
end

function readcpp(io::IO, ::Type{TagSet}; kwargs...)
  format = get(kwargs, :format, "v3")
  ts = TagSet()
  if format == "v3"
    mstore = empty_storage(MTagSetStorage{IntTag,4})
    ntags = 0
    for n in 1:4
      t = readcpp(io, Tag; kwargs...)
      if t != Tag()
        ntags = _addtag_ordered!(mstore, ntags, IntTag(t))
      end
    end
    ts = TagSet(TagSetStorage(mstore), ntags)
  else
    throw(ArgumentError("read TagSet: format=$format not supported"))
  end
  return ts
end

function HDF5.write(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, T::TagSet)
  g = create_group(parent, name)
  attributes(g)["type"] = "TagSet"
  attributes(g)["version"] = 1
  return write(g, "tags", tagstring(T))
end

function HDF5.read(
  parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{TagSet}
)
  g = open_group(parent, name)
  if read(attributes(g)["type"]) != "TagSet"
    error("HDF5 group '$name' does not contain TagSet data")
  end
  tstring = read(g, "tags")
  return TagSet(tstring)
end
