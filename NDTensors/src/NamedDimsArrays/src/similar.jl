# `similar`
# TODO: Can these definitions be compressed into a more minimal set?
function similar_parent(na::AbstractNamedDimsArray, args...)
  return named(similar(unname(na), args...), dimnames(na))
end
function Base.similar(na::AbstractNamedDimsArray, elt::Type, dims::Tuple{Vararg{Int64}})
  return similar_parent(na, elt, dims)
end
function Base.similar(
  na::AbstractNamedDimsArray, elt::Type, dims::Tuple{Integer,Vararg{Integer}}
)
  return similar_parent(na, elt, dims)
end
function Base.similar(
  na::AbstractNamedDimsArray,
  elt::Type,
  dims::Tuple{Union{Integer,Base.OneTo},Vararg{Union{Integer,Base.OneTo}}},
)
  return similar_parent(na, elt, dims)
end
function Base.similar(
  na::AbstractNamedDimsArray, elt::Type, dims::Union{Integer,AbstractUnitRange}...
)
  return similar_parent(na, elt, dims...)
end
Base.similar(na::AbstractNamedDimsArray, elt::Type) = similar_parent(na, elt)
Base.similar(na::AbstractNamedDimsArray, dims::Tuple) = similar_parent(na, dims)
function Base.similar(na::AbstractNamedDimsArray, dims::Union{Integer,AbstractUnitRange}...)
  return similar_parent(na, dims...)
end
Base.similar(na::AbstractNamedDimsArray) = similar_parent(na)
