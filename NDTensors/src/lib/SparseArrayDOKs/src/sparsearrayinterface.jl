using Dictionaries: Dictionary

# Also look into:
# https://juliaarrays.github.io/ArrayInterface.jl/stable/sparsearrays/

# Required `SparseArrayInterface` interface.
# https://github.com/Jutho/SparseArrayKit.jl interface functions
nonzero_keys(a::AbstractArray) = error("Not implemented")
nonzero_values(a::AbstractArray) = error("Not implemented")
nonzero_pairs(a::AbstractArray) = error("Not implemented")

# A dictionary-like structure
# TODO: Rename `nonzeros`, `structural_nonzeros`, etc.?
nonzero_structure(a::AbstractArray) = error("Not implemented")

# Derived `SparseArrayInterface` interface.
nonzero_length(a::AbstractArray) = length(nonzero_keys(a))
is_structural_nonzero(a::AbstractArray, I) = I ∈ nonzero_keys(a)

# Overload if zero value is index dependent or
# doesn't match element type.
getindex_nonzero(a::AbstractArray, I) = nonzero_structure(a)[I]
getindex_zero(a::AbstractArray, I) = zero(eltype(a))
function setindex_zero!(a::AbstractArray, value, I)
  # TODO: This may need to be modified.
  nonzero_structure(a)[I] = value
  return a
end
function setindex_nonzero!(a::AbstractArray, value, I)
  nonzero_structure(a)[I] = value
  return a
end

struct Zero end
(::Zero)(type, I) = zero(type)

default_zero() = Zero() # (eltype, I) -> zero(eltype)
default_keytype(ndims::Int) = CartesianIndex{ndims}
default_data(type::Type, ndims::Int) = Dictionary{default_keytype(ndims),type}()
