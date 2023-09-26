module TagSets
using Dictionaries

## import Dictionaries:
##   istokenizable,
##   tokentype,
##   iteratetoken,
##   iteratetoken_reverse,
##   gettoken,
##   gettokenvalue,
##   isinsertable,
##   gettoken!,
##   empty_type,
##   deletetoken!,
##   randtoken

using Base: @propagate_inbounds

## using NDTensors.SmallVectors
## using InlineStrings

## using NDTensors.SmallVectors: AbstractSmallVector, buffer

# A sorted collection of unique tags of type `T`.
# Add `skipchars` (see `skipmissing`) and `delim` for delimiter?
# https://docs.julialang.org/en/v1/base/strings/#Base.strip
# https://docs.julialang.org/en/v1/stdlib/DelimitedFiles/#Delimited-Files
# Add a `Bool` param for bounds checking/ignoring overflow/spillover?
# TODO: Make `S` a first argument, hardcode `SmallVector` storage?
# https://juliacollections.github.io/DataStructures.jl/v0.9/sorted_containers.html
# https://github.com/JeffreySarnoff/SortingNetworks.jl
# https://github.com/vvjn/MergeSorted.jl
# https://bkamins.github.io/julialang/2023/08/25/infiltrate.html
# https://github.com/Jutho/TensorKit.jl/blob/master/src/auxiliary/dicts.jl
# https://github.com/tpapp/SortedVectors.jl
# https://discourse.julialang.org/t/special-purpose-subtypes-of-arrays/20327
# https://discourse.julialang.org/t/all-the-ways-to-group-reduce-sorted-vectors-ideas/45239
# https://discourse.julialang.org/t/sorting-a-vector-of-fixed-size/71766
struct TagSet{T,D<:AbstractIndices{T}} <: AbstractIndices{T}
  data::D
end

TagSet(vec::AbstractVector) = TagSet(Indices(vec))

# Field accessors
Base.parent(tags::TagSet) = getfield(tags, :data)

# AbstractIndices interface
@propagate_inbounds function Base.iterate(tags::TagSet, state...)
  return iterate(parent(tags), state...)
end

# `I` is needed to avoid ambiguity error.
Base.in(tag::I, tags::TagSet{I}) where {I} = in(tag, parent(tags))
Base.IteratorSize(::TagSet) = Base.HasLength()
Base.length(tags::TagSet) = length(parent(tags))

Dictionaries.istokenizable(i::TagSet) = true
Dictionaries.tokentype(::TagSet) = Int
@inline Dictionaries.iteratetoken(inds::TagSet, s...) = iterate(parent(inds), s...)
@inline function Dictionaries.iteratetoken_reverse(inds::TagSet)
  return iteratetoken_reverse(parent(inds))
end
@inline function Dictionaries.iteratetoken_reverse(inds::TagSet, t)
  return iteratetoken_reverse(parent(inds), t)
end

@inline function Dictionaries.gettoken(inds::TagSet, i)
  return gettoken(parent(inds), i)
end
@propagate_inbounds Dictionaries.gettokenvalue(inds::TagSet, x) = gettokenvalue(parent(inds), x)

Dictionaries.isinsertable(tags::TagSet) = true # Need an array trait here...

# Specify `I` to fix ambiguity error.
@inline function Dictionaries.gettoken!(tags::TagSet{I}, i::I, values=()) where {I}
  return gettoken!(parent(tags), i, values)
end

@inline function Dictionaries.deletetoken!(tags::TagSet, x, values=())
  deletetoken!(parent(tags), x, values)
  return tags
end

function Base.empty!(inds::TagSet, values=())
  empty!(parent(inds))
  return inds
end

Dictionaries.empty_type(::Type{TagSet{I,D}}, ::Type{I}) where {I,D} = TagSet{I,D}

# Not defined to be part of the `AbstractIndices` interface,
# but seems to be needed.
function Base.filter!(pred, inds::TagSet)
  filter!(pred, parent(inds))
  return inds
end

function Base.copy(tags::TagSet, eltype::Type)
  return TagSet(copy(parent(tags), eltype))
end

# TagSet interface
addtags(tags::TagSet, items) = union(tags, items)
removetags(tags::TagSet, items) = setdiff(tags, items)
function replacetags(tags::TagSet, rem, add)
  remtags = setdiff(tags, rem)
  if length(tags) ≠ length(remtags) + length(rem)
    # Not all are removed, no replacement
    return tags
  end
  return union(remtags, add)
end

## valparam(::Val{T}) where {T} = T
## function default_datatype(tagset_type::Type{<:TagSet}, maxlengthval::Val)
##   return SmallVector{valparam(maxlengthval),eltype(tagset_type)}
## end
## function default_datatype(tagset::TagSet, maxlengthval::Val)
##   return default_datatype(typeof(tagset), maxlengtval, eltype)
## end

## # Assumes sorted, rename `tagsetsorted`? External should sort?
## function TagSet{T}(maxlengthval::Val, data::AbstractVector) where {T}
##   newdata = convert(default_datatype(TagSet{T}, maxlengthval), data)
##   # newdata = unique(sort(newdata))
##   return TagSet{T,typeof(newdata)}(newdata)
## end

## # undef constructors (delete, only allow empty constructor)
## ## TagSet{T,D}(undef::UndefInitializer, length::Integer) where {T,D} = TagSet{T,D}(D(undef, length))
## ## TagSet(datatype::Type{<:AbstractVector}, ::UndefInitializer, length::Integer) = TagSet{eltype(datatype),datatype}(undef, length)

## # empty constructors
## TagSet{T,D}() where {T,D} = TagSet{T,D}(D())
## TagSet(datatype::Type{<:AbstractVector}) = TagSet{eltype(datatype),datatype}()

## ## default_datatype(tagset_type::Type{TagSet{S,T}}) where {S,T} = SmallVector{S,T}
## ## default_datatype(tagset::TagSet) = default_datatype(typeof(tagset))

## ## function TagSet{S,T}(data::AbstractVector) where {S,T}
## ##   newdata = convert(default_datatype(TagSet{S,T}), data)
## ##   return TagSet{S,T,typeof(newdata)}(newdata)
## ## end

## ## TagSet{S}(data::AbstractVector) where {S} = TagSet{S,eltype(data)}(data)

## # TODO: Make an empty constructor for `SmallString`.
## ## (tagset_type::Type{TagSet{S,T}})() where {S,T} = TagSet{S,T}(default_datatype(tagset_type)([]))

## # Required interface
## Base.size(vec::TagSet) = size(vec.data)
## SmallVectors.buffer(vec::TagSet) = buffer(vec.data)
## @inline function Base.getindex(vec::TagSet, index::Integer)
##   @boundscheck checkbounds(vec, index)
##   return @inbounds buffer(vec)[index]
## end

## # Specialized constructor
## function MSmallVector{S,T}(vec::TagSet) where {S,T}
##   return MSmallVector{S,T}(vec.data)
## end

## # Optimization, default uses `similar`.
## # TODO: See if this can be improved.
## Base.copymutable(vec::TagSet) = MSmallVector(vec)

## # TagSet functions
## using NDTensors.SmallVectors: insert, deleteat, insertsortedunique, insertsortedunique!
## function addtag(tagset::TagSet, tag::AbstractString)
##   return insertsortedunique(tagset, tag)
## end
## function removetag(tagset::TagSet, tag::AbstractString)
##   # TODO: Defined `deletesorted`.
##   return deleteat(tagset, searchsortedfirst(tagset, tag))
## end
## function addtags(tagset::TagSet, tags::AbstractVector)
##   # unionsorted[!]
##   for tag in tags
##     tagset = addtag(tagset, tag)
##   end
##   return tagset

##   ## # unionsorted[!]
##   ## mvec = Base.copymutable(tagset)
##   ## for tag in tags
##   ##   # tagset = addtag(tagset, tag)
##   ##   insertsortedunique!(mvec, tag)
##   ## end
##   ## return typeof(tagset)(mvec)
## end
## function removetags(tagset::TagSet, tags::AbstractVector)
##   # setdiffsorted[!]
##   mvec = Base.copymutable(tagset)
##   for tag in tags
##     # deletesortedunique[!]
##     # tagset = removetag!(tagset, tag)
##     r = searchsorted(mvec, tag)
##     if length(r) > 0 # !isempty(r)
##       # Check `length(r) == 1` for uniqueness.
##       deleteat!(mvec, first(r))
##     end
##   end
##   return typeof(tagset)(mvec)
## end
end

## # Functionality:
## # hastag (insorted), hastags (insorted), not, replacetags, tagstring, commontags, readcpp, HDF5.write, HDF5.read
## # TagSet(::String), convert(::Type{TagSet}, String) (maybe?)
## # String(::TagSet) (tagstring?)
## # Custom ignore char (' ') and deliminator (',').
## # Strict tags to ignore overflow/spillover of tag length and/or number of tags.
## 
## ## for sz in (1, 4, 8, 16, 32, 64, 128, 256)
## ##     nm = Symbol(:String, max(1, sz - 1))
## ##     nma = Symbol(:TagSet, max(1, sz - 1))
## ##     macro_nm = Symbol(:ts, max(1, sz - 1), :_str)
## ##     at_macro_nm = Symbol("@", macro_nm)
## ##     @eval begin
## ##     end
## ## end
## 
## # Example
## using BenchmarkTools
## using ProfileView
## 
## using ITensors: ITensors
## 
## # String1, String3, String7, String15, String31, String63, String127, String255
## # inline1"", inline3"", etc.
## # tagset1"", tagset3"", etc.
## # ts1"", ts3"", etc.
## function main(; string_type=String63)
##   tagset_data_type = SmallVector{20,string_type}
##   ts = TagSet(tagset_data_type)
##   @show ts
## 
##   ts = addtag(ts, "x")
##   @show ts
##   @btime addtag($ts, $(eltype(ts)("x")))
## 
##   ts = addtag(ts, "y")
##   @show ts
##   @btime addtag($ts, $(eltype(ts)("y")))
## 
##   ts = removetag(ts, "y")
##   @show ts
##   @btime removetag($ts, $(eltype(ts)("y")))
## 
##   ts = addtags(ts, ["a", "z"])
##   @show ts
##   @btime addtags($ts, $(typeof(ts)(["a", "z"])))
## 
##   ts′ = removetags(ts, ["a", "z"])
##   @show ts′
##   @btime removetags($ts, $(typeof(ts)(["a", "z"])))
## 
##   println("ITensor version of addtags")
##   @btime ITensors.addtags($(ITensors.TagSet("x,y")), $(ITensors.TagSet("a,z")))
## 
##   println("Equality")
##   println("SmallVectors")
##   tags = typeof(ts)(["x", "y"])
##   @btime $tags == $tags
##   println("ITensors")
##   @btime $(ITensors.TagSet("x,y")) == $(ITensors.TagSet("x,y"))
## end
