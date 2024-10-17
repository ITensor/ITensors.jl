const AbstractSmallSet{T} = SortedSet{T,<:AbstractSmallVector{T}}
const SmallSet{S,T} = SortedSet{T,SmallVector{S,T}}
const MSmallSet{S,T} = SortedSet{T,MSmallVector{S,T}}

# Specialized constructors
@propagate_inbounds SmallSet{S}(; kwargs...) where {S} = SmallSet{S}([]; kwargs...)
@propagate_inbounds SmallSet{S}(iter; kwargs...) where {S} = SmallSet{S}(
  collect(iter); kwargs...
)
@propagate_inbounds SmallSet{S}(a::AbstractArray{I}; kwargs...) where {S,I} = SmallSet{S,I}(
  a; kwargs...
)

@propagate_inbounds MSmallSet{S}(; kwargs...) where {S} = MSmallSet{S}([]; kwargs...)
@propagate_inbounds MSmallSet{S}(iter; kwargs...) where {S} = MSmallSet{S}(
  collect(iter); kwargs...
)
@propagate_inbounds MSmallSet{S}(a::AbstractArray{I}; kwargs...) where {S,I} = MSmallSet{
  S,I
}(
  a; kwargs...
)
