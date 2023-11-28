# TODO: Rename the preserve zero case to `map_nonzeros!`,
# move to `sparsearrayinterface.jl`.
function Base.map!(f, a_dest::AbstractArray, as::SparseArrayDOK...)
  # TODO: Handle nonzero case, fill all values.
  # Also, handle arrays of arrays better, using `getindex_zero(a, first(eachindex(a)))`.
  @assert iszero(f(map(a -> getindex_zero(a, first(eachindex(a))), as)...))
  Is = union(nonzero_keys.(as)...)
  for I in Is
    a_dest[I] = f(map(a -> a[I], as)...)
  end
  return a_dest
end
