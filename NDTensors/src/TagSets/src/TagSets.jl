module TagSets
using Dictionaries
using ..SortedSets

export TagSet, addtags, removetags, replacetags

# A sorted collection of unique tags of type `T`.
struct TagSet{T,D<:AbstractIndices{T}} <: AbstractWrappedIndices{T,D}
  data::D
end

TagSet{T,D}(vec::AbstractVector) where {T,D<:AbstractIndices{T}} = TagSet{T,D}(D(vec))
TagSet{T,D}() where {T,D<:AbstractIndices{T}} = TagSet{T,D}(D())

# Defaults to Indices if unspecified.
TagSet(vec::AbstractVector) = TagSet(Indices(vec))

# Field accessors
@inline Base.parent(tags::TagSet) = getfield(tags, :data)

# AbstractWrappedSet interface.
@inline SortedSets.rewrap(vec::TagSet, data) = TagSet(data)

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
