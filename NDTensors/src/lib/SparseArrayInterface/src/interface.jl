# Also look into:
# https://juliaarrays.github.io/ArrayInterface.jl/stable/sparsearrays/

# Minimal interface
# Data structure storing the nonzero values
nonzeros(a::AbstractArray) = a

# Minimal interface
# Map an index in the nonzero data to a CartesianIndex of the
# outer array.
nonzero_index_to_index(a::AbstractArray, Inz) = Inz

# Minimal interface
# Map a `CartesianIndex` to an index/key into the nonzero data structure
# returned by `nonzeros`.
# Return `nothing` if the index corresponds to a structural zero value.
index_to_nonzero_index(a::AbstractArray{<:Any,N}, I::CartesianIndex{N}) where {N} = I
