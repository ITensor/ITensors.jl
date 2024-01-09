# Also look into:
# https://juliaarrays.github.io/ArrayInterface.jl/stable/sparsearrays/

# Minimal interface
# Data structure of the stored (generally nonzero) values
sparse_storage(a::AbstractArray) = error("Not implemented")

sparse_storage(a::Array) = a

# Minimal interface
# Map an index into the stored data to a CartesianIndex of the
# outer array.
storage_index_to_index(a::AbstractArray, I) = I

# Minimal interface
# Map a `CartesianIndex` to an index/key into the nonzero data structure
# returned by `storage`.
# Return `nothing` if the index corresponds to a structural zero (unstored) value.
index_to_storage_index(a::AbstractArray{<:Any,N}, I::CartesianIndex{N}) where {N} = I
