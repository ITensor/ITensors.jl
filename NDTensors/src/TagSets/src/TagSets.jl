module TagSets
using Dictionaries
using ..SmallVectors
using ..SortedSets

using Base: @propagate_inbounds

export TagSet, SmallTagSet, addtags, removetags, replacetags, commontags, noncommontags

# A sorted collection of unique tags of type `T`.
struct TagSet{T,D<:AbstractIndices{T}} <: AbstractWrappedIndices{T,D}
  data::D
end

TagSet{T}(data::D) where {T,D<:AbstractIndices{T}} = TagSet{T,D}(data)

TagSet{T,D}(vec::AbstractVector) where {T,D<:AbstractIndices{T}} = TagSet{T,D}(D(vec))
TagSet{T,D}() where {T,D<:AbstractIndices{T}} = TagSet{T,D}(D())

# Defaults to Indices if unspecified.
default_data_type() = Indices{String}
TagSet(vec::AbstractVector) = TagSet(default_data_type()(vec))

# Constructor from string
default_delim() = ','
@inline function TagSet(str::AbstractString; delim=default_delim())
  return TagSet(default_data_type(), str)
end
@inline function TagSet(
  ::Type{D}, str::AbstractString; delim=default_delim()
) where {T,D<:AbstractIndices{T}}
  return TagSet{T,D}(str)
end
@inline function TagSet{T,D}(
  str::AbstractString; delim=default_delim()
) where {T,D<:AbstractIndices{T}}
  return TagSet{T,D}(split(str, delim))
end

const SmallTagSet{S,T} = TagSet{T,SmallSet{S,T}}
@propagate_inbounds SmallTagSet{S}(; kwargs...) where {S} = SmallTagSet{S}([]; kwargs...)
@propagate_inbounds SmallTagSet{S}(iter; kwargs...) where {S} =
  SmallTagSet{S}(collect(iter); kwargs...)
@propagate_inbounds SmallTagSet{S}(a::AbstractArray{I}; kwargs...) where {S,I} =
  SmallTagSet{S,I}(a; kwargs...)
# Specialized `SmallSet{S,T} = SortedSet{T,SmallVector{S,T}}` constructor
function SmallTagSet{S,T}(str::AbstractString; delim=default_delim()) where {S,T}
  # TODO: Optimize for `SmallSet`.
  return SmallTagSet{S,T}(split(str, delim))
end

# Field accessors
Base.parent(tags::TagSet) = getfield(tags, :data)

# AbstractWrappedSet interface.
# Specialized version when they are the same data type is faster.
@inline SortedSets.rewrap(vec::TagSet{T,D}, data::D) where {T,D<:AbstractIndices{T}} =
  TagSet{T,D}(data)
@inline SortedSets.rewrap(vec::TagSet{T,D}, data) where {T,D<:AbstractIndices{T}} =
  TagSet{T,D}(data)

# TagSet interface
addtags(tags::TagSet, items) = union(tags, items)
removetags(tags::TagSet, items) = setdiff(tags, items)
commontags(tags::TagSet, items) = intersect(tags, items)
noncommontags(tags::TagSet, items) = symdiff(tags, items)
function replacetags(tags::TagSet, rem, add)
  remtags = setdiff(tags, rem)
  if length(tags) ≠ length(remtags) + length(rem)
    # Not all are removed, no replacement
    return tags
  end
  return union(remtags, add)
end

end
