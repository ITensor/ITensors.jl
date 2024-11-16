# Also look into:
# https://juliaarrays.github.io/ArrayInterface.jl/stable/sparsearrays/

# Minimal sparse array interface.
# Data structure of the stored (generally nonzero) values.
# By default assume it is dense, so all values are stored.
sparse_storage(a::AbstractArray) = a

# Minimal sparse array interface.
# Map an index into the stored data to a CartesianIndex of the
# outer array.
storage_index_to_index(a::AbstractArray, I) = I

# Minimal interface
# Map a `CartesianIndex` to an index/key into the nonzero data structure
# returned by `storage`.
# Return `nothing` if the index corresponds to a structural zero (unstored) value.
index_to_storage_index(a::AbstractArray{<:Any,N}, I::CartesianIndex{N}) where {N} = I
