abstract type AbstractNamedUnitRange{T,Value<:AbstractUnitRange{T},Name} <:
              AbstractUnitRange{T} end

# Required interface
unname(::AbstractNamedUnitRange) = error("Not implemented")
name(::AbstractNamedUnitRange) = error("Not implemented")

# Traits
isnamed(::AbstractNamedUnitRange) = true

# Unit range
Base.first(i::AbstractNamedUnitRange) = first(unname(i))
Base.last(i::AbstractNamedUnitRange) = last(unname(i))
Base.length(i::AbstractNamedUnitRange) = named(length(unname(i)), name(i))

# TODO: Use `isnamed` trait?
dimnames(a::Tuple{Vararg{AbstractNamedUnitRange}}) = name.(a)

unname(a::Tuple{Vararg{AbstractNamedUnitRange}}) = unname.(a)
unname(a::Tuple{Vararg{AbstractNamedUnitRange}}, names) = unname(align(a, names))

function named(as::Tuple{Vararg{AbstractUnitRange}}, names)
  return ntuple(j -> named(as[j], names[j]), length(as))
end

function get_name_perm(a::Tuple{Vararg{AbstractNamedUnitRange}}, names::Tuple)
  return getperm(dimnames(a), names)
end

# Permute into a certain order.
# align(a, (:j, :k, :i))
function align(a::Tuple{Vararg{AbstractNamedUnitRange}}, names)
  perm = get_name_perm(a, names)
  return map(j -> a[j], perm)
end
