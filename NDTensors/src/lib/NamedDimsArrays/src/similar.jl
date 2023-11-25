# `similar`

# Preserve the names
Base.similar(na::AbstractNamedDimsArray) = named(similar(unname(na)), dimnames(na))
function Base.similar(na::AbstractNamedDimsArray, elt::Type)
  return named(similar(unname(na), elt), dimnames(na))
end

# Remove the names
# TODO: Make versions taking `NamedUnitRange` and `NamedInt`.
function Base.similar(na::AbstractNamedDimsArray, elt::Type, dims::Tuple{Vararg{Int64}})
  return similar(unname(na), elt, dims)
end

# Remove the names
# TODO: Make versions taking `NamedUnitRange` and `NamedInt`.
function Base.similar(
  na::AbstractNamedDimsArray, elt::Type, dims::Tuple{Integer,Vararg{Integer}}
)
  return similar(unname(na), elt, dims)
end

# Remove the names
# TODO: Make versions taking `NamedUnitRange` and `NamedInt`.
function Base.similar(
  na::AbstractNamedDimsArray,
  elt::Type,
  dims::Tuple{Union{Integer,Base.OneTo},Vararg{Union{Integer,Base.OneTo}}},
)
  return similar(unname(na), elt, dims)
end

# Remove the names
# TODO: Make versions taking `NamedUnitRange` and `NamedInt`.
function Base.similar(
  na::AbstractNamedDimsArray, elt::Type, dims::Union{Integer,AbstractUnitRange}...
)
  return similar(unname(na), elt, dims...)
end

# Remove the names
# TODO: Make versions taking `NamedUnitRange` and `NamedInt`.
Base.similar(na::AbstractNamedDimsArray, dims::Tuple) = similar(unname(na), dims)

# Remove the names
# TODO: Make versions taking `NamedUnitRange` and `NamedInt`.
function Base.similar(na::AbstractNamedDimsArray, dims::Union{Integer,AbstractUnitRange}...)
  return similar(unname(na), dims...)
end
