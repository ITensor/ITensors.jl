# Optional interface.
# Access a zero value.
function getindex_notstored(a::AbstractArray, I)
  return zero(eltype(a))
end

# Optional interface.
# Insert a new value into a location
# where there is not a stored value.
# Some types (like `Diagonal`) may not support this.
function setindex_notstored!(a::AbstractArray, value, I)
  return throw(ArgumentError("Can't set nonzero values of $(typeof(a))."))
end

# Optional interface.
# Iterates over the indices of `a` where there are stored values.
# Can overload for faster iteration when there is more structure,
# for example with DiagonalArrays.
function stored_indices(a::AbstractArray)
  return Iterators.map(Inz -> storage_index_to_index(a, Inz), storage_indices(a))
end

# Empty the sparse storage if possible.
# Array types should overload `Base.dataids` to opt-in
# to aliasing detection with `Base.mightalias`
# to avoid emptying an input array in the case of `sparse_map!`.
# `empty_storage!` is used to zero out the output array.
# See also `Base.unalias` and `Base.unaliascopy`.
empty_storage!(a::AbstractArray) = a

# Overload
function sparse_similar(a::AbstractArray, elt::Type, dims::Tuple{Vararg{Int}})
  return similar(a, elt, dims)
end
