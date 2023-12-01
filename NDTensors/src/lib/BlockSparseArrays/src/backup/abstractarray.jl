# Dense fallbacks for sparse functions
nonzero_keys(a::AbstractArray) = keys(a)
map_nonzeros(f, a::AbstractArray) = map(f, a)
map_nonzeros!(f, a_dest::AbstractArray, a_src::AbstractArray) = map!(f, a_dest, a_src)
