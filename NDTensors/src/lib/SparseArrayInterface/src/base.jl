# TODO: Use `sparse_mapreduce`?
function sparse_iszero(a::AbstractArray)
  iszero(nstored(a)) && return true
  return iszero(storage(a))
end

# TODO: Use `sparse_mapreduce`?
function sparse_isreal(a::AbstractArray)
  iszero(nstored(a)) && return true
  return isreal(storage(a))
end

function sparse_fill!(a::AbstractArray, x)
  sparse_map!(Returns(x), a, a)
  return a
end

function sparse_zero(a::AbstractArray)
  # Need to overload `similar` for custom types
  a = similar(a)
  # TODO: Use custom zero value?
  sparse_fill!(a, zero(eltype(a)))
  return a
end

# TODO: Make `sparse_one!`?
function sparse_one(a::AbstractMatrix)
  m, n = size(a)
  @assert m == n
  a = zero(a)
  # TODO: Use `diagindices` from `DiagonalArrays`?
  for i in 1:m
    a[i, i] = one(eltype(a))
  end
  return a
end

# TODO: Use `sparse_mapreduce`?
function sparse_isequal(a1::AbstractArray, a2::AbstractArray)
  Is = collect(stored_indices(a1))
  intersect!(Is, stored_indices(a2))
  if !(length(Is) == nstored(a1) == nstored(a2))
    return false
  end
  for I in Is
    a1[I] == a2[I] || return false
  end
  return true
end

function sparse_reshape(a::AbstractArray, dims)
  @assert length(a) == prod(dims)
  a_reshaped = similar(a, dims)
  sparse_fill!(a_reshaped, zero(eltype(a)))
  linear_inds = LinearIndices(a)
  new_cartesian_inds = CartesianIndices(dims)
  for I in stored_indices(a)
    a_reshaped[new_cartesian_inds[linear_inds[I]]] = a[I]
  end
  return a_reshaped
end

## function sparse_mapreduce(f, op, as::AbstractArray...)
##   # TODO: Make more general.
##   @assert iszero(mapreduce(f, op, map(a -> getindex_zero(a, first(eachindex(a))), as)...))
##   # TODO: Create a `promote_stored_indices`, for example to help
##   # with specialized fast indexing for `DiagonalArrays`.
##   Is = union(stored_indices.(as)...)
##   for I in Is
##     a_dest[I] = f(map(a -> a[I], as)...)
##   end
##   return reduce(op, a_dest)
## end
