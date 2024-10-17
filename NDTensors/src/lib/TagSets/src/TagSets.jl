module TagSets
using Dictionaries
using ..SmallVectors
using ..SortedSets

using Base: @propagate_inbounds

export TagSet,
  SmallTagSet, MSmallTagSet, addtags, removetags, replacetags, commontags, noncommontags

# A sorted collection of unique tags of type `T`.
struct TagSet{T,D<:AbstractIndices{T}} <: AbstractWrappedSet{T,D}
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

for (SetTyp, TagSetTyp) in ((:SmallSet, :SmallTagSet), (:MSmallSet, :MSmallTagSet))
  @eval begin
    const $TagSetTyp{S,T,Order} = TagSet{T,$SetTyp{S,T,Order}}
    @propagate_inbounds function $TagSetTyp{S,I}(a::AbstractArray; kwargs...) where {S,I}
      return TagSet($SetTyp{S,I}(a; kwargs...))
    end
    @propagate_inbounds $TagSetTyp{S}(; kwargs...) where {S} = $TagSetTyp{S}([]; kwargs...)
    @propagate_inbounds $TagSetTyp{S}(iter; kwargs...) where {S} = $TagSetTyp{S}(
      collect(iter); kwargs...
    )
    @propagate_inbounds $TagSetTyp{S}(a::AbstractArray{I}; kwargs...) where {S,I} = $TagSetTyp{
      S,I
    }(
      a; kwargs...
    )
    # Strings get split by a deliminator.
    function $TagSetTyp{S}(str::T; kwargs...) where {S,T<:AbstractString}
      return $TagSetTyp{S,T}(str, kwargs...)
    end
    # Strings get split by a deliminator.
    function $TagSetTyp{S,T}(
      str::AbstractString; delim=default_delim(), kwargs...
    ) where {S,T}
      # TODO: Optimize for `SmallSet`.
      return $TagSetTyp{S,T}(split(str, delim); kwargs...)
    end
  end
end

# Field accessors
Base.parent(set::TagSet) = getfield(set, :data)

# AbstractWrappedSet interface.
# Specialized version when they are the same data type is faster.
@inline SortedSets.rewrap(::TagSet{T,D}, data::D) where {T,D<:AbstractIndices{T}} = TagSet{
  T,D
}(
  data
)
@inline SortedSets.rewrap(::TagSet, data) = TagSet(data)

# TagSet interface
addtags(set::TagSet, items) = union(set, items)
removetags(set::TagSet, items) = setdiff(set, items)
commontags(set::TagSet, items) = intersect(set, items)
noncommontags(set::TagSet, items) = symdiff(set, items)
function replacetags(set::TagSet, rem, add)
  remtags = setdiff(set, rem)
  if length(set) ≠ length(remtags) + length(rem)
    # Not all are removed, no replacement
    return set
  end
  return union(remtags, add)
end

end
