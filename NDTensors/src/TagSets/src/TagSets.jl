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
end
