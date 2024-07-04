# Optional interface.

# Function for computing unstored zero values.
getindex_zero_function(::AbstractArray) = Zero()

# Change the function for computing unstored values
set_getindex_zero_function(a::AbstractArray, f) = error("Not implemented")

function getindex_notstored(a::AbstractArray, I)
  return getindex_zero_function(a)(a, I)
end

# Optional interface.
# Insert a new value into a location
# where there is not a stored value.
# Some types (like `Diagonal`) may not support this.
function setindex_notstored!(a::AbstractArray, value, I)
  iszero(value) && return a
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
# `dropall!` is used to zero out the output array.
# See also `Base.unalias` and `Base.unaliascopy`.
# Interface is inspired by Base `SparseArrays.droptol!`
# and `SparseArrays.dropzeros!`, and is like
# `dropall!(a) = SparseArrays.droptol!(a, Inf)`.
dropall!(a::AbstractArray) = a

# Overload
function sparse_similar(a::AbstractArray, elt::Type, dims::Tuple{Vararg{Int}})
  return similar(a, elt, dims)
end
