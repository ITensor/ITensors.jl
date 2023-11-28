# TODO: Use `sparse_mapreduce`?
function sparse_iszero(a::AbstractArray)
  iszero(nonzero_length(a)) && return true
  return iszero(nonzeros(a))
end

# TODO: Use `sparse_mapreduce`?
function sparse_isreal(a::AbstractArray)
  iszero(nonzero_length(a)) && return true
  return isreal(nonzeros(a))
end

function fill_nonzero!(a::AbstractArray, x)
  for I in eachindex(a)
    a[I] = x
  end
  return a
end

function fill_zero!(a::AbstractArray, x)
  fill!(nonzeros(a), x)
  return a
end

function sparse_fill!(a::AbstractArray, x)
  if !iszero(x)
    # TODO: Reverse naming convention?
    fill_nonzero!(a, x)
  else
    fill_zero!(a, x)
  end
  return a
end

function sparse_zero(a::AbstractArray)
  # Need to overload `similar` for custom types
  a = similar(a)
  sparse_fill!(a, zero(eltype(a)))
  return a
end

## function sparse_zero(a::AbstractArray, elt::Type)
##   return sparse_zero(a, elt, size(a))
## end
## 
## function sparse_zero(a::AbstractArray, dims::Tuple)
##   return sparse_zero(a, eltype(a), dims)
## end
## 
## function sparse_zero(a::AbstractArray)
##   return sparse_zero(a, eltype(a), size(a))
## end

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
  Is = collect(nonzero_indices(a1))
  intersect!(Is, nonzero_indices(a2))
  if !(length(Is) == nonzero_length(a1) == nonzero_length(a2))
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
  for I in nonzero_indices(a)
    a_reshaped[new_cartesian_inds[linear_inds[I]]] = a[I]
  end
  return a_reshaped
end

## function sparse_mapreduce(f, op, as::AbstractArray...)
##   # TODO: Make more general.
##   @assert iszero(mapreduce(f, op, map(a -> getindex_zero(a, first(eachindex(a))), as)...))
##   # TODO: Create a `promote_nonzero_indices`, for example to help
##   # with specialized fast indexing for `DiagonalArrays`.
##   Is = union(nonzero_indices.(as)...)
##   for I in Is
##     a_dest[I] = f(map(a -> a[I], as)...)
##   end
##   return reduce(op, a_dest)
## end
