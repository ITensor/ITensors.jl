# Test if the function preserves zero values and therefore
# preserves the sparsity structure.
function preserves_zero(f, a_dest, as...)
  return iszero(f(map(a -> sparse_getindex(a, NotStoredIndex(first(eachindex(a)))), as)...))
end

# Map a subset of indices
function sparse_map_indices!(f, a_dest::AbstractArray, indices, as::AbstractArray...)
  for I in indices
    a_dest[I] = f(map(a -> a[I], as)...)
  end
  return a_dest
end

# Overload for custom `stored_indices` types.
function promote_indices(I1, I2)
  return union(I1, I2)
end

function promote_indices(I1, I2, Is...)
  return promote_indices(promote_indices(I1, I2), Is...)
end

# Base case
promote_indices(I) = I

function promote_stored_indices(as::AbstractArray...)
  return promote_indices(stored_indices.(as)...)
end

function sparse_map_stored!(f, a_dest::AbstractArray, as::AbstractArray...)
  # TODO: Create a `promote_stored_indices`, for example to help
  # with specialized fast indexing for `DiagonalArrays`.
  Is = promote_stored_indices(as...)
  sparse_map_indices!(f, a_dest, Is, as...)
  return a_dest
end

# Handle nonzero case, fill all values.
function sparse_map_all!(f, a_dest::AbstractArray, as::AbstractArray...)
  Is = eachindex(a_dest)
  sparse_map_indices!(f, a_dest, Is, as...)
  return a_dest
end

function sparse_map!(f, a_dest::AbstractArray, as::AbstractArray...)
  @assert allequal(axes.((a_dest, as...)))
  if preserves_zero(f, a_dest, as...)
    sparse_map_stored!(f, a_dest, as...)
  else
    sparse_map_all!(f, a_dest, as...)
  end
  return a_dest
end
