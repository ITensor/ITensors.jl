# TODO: Rename the preserve zero case to `map_nonzeros!`.
function sparse_map!(f, a_dest::AbstractArray, as::AbstractArray...)
  # TODO: Handle nonzero case, fill all values.
  @assert iszero(f(map(a -> getindex_zero(a, first(eachindex(a))), as)...))
  # TODO: Create a `promote_nonzero_indices`, for example to help
  # with specialized fast indexing for `DiagonalArrays`.
  Is = union(nonzero_indices.(as)...)
  for I in Is
    a_dest[I] = f(map(a -> a[I], as)...)
  end
  return a_dest
end
