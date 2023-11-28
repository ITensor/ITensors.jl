function sparse_reduce(op, a::AbstractArray; kwargs...)
  return sparse_mapreduce(identity, op, a; kwargs...)
end

function sparse_all(a::AbstractArray)
  return sparse_reduce(&, a; init=true)
end

function sparse_all(f, a::AbstractArray)
  return sparse_mapreduce(f, &, a; init=true)
end

function sparse_iszero(a::AbstractArray)
  return sparse_all(iszero, a)
end

function sparse_isreal(a::AbstractArray)
  return sparse_all(isreal, a)
end

# This is equivalent to:
#
# sparse_map!(Returns(x), a, a)
#
# but we don't use that here since `sparse_fill!`
# is used inside of `sparse_map!`.
function sparse_fill!(a::AbstractArray, x)
  if iszero(x)
    empty_storage!(a)
  end
  fill!(storage(a), x)
  return a
end

# This could just call `sparse_fill!`
# but it avoids a zero construction and check.
function sparse_zero!(a::AbstractArray)
  empty_storage!(a)
  fill!(storage(a), zero(eltype(a)))
  return a
end

# TODO: Make `sparse_zero!`?
function sparse_zero(a::AbstractArray)
  # Need to overload `similar` for custom types
  a = similar(a)
  # TODO: Use custom zero value?
  sparse_fill!(a, zero(eltype(a)))
  return a
end

# TODO: Is this a good definition?
function sparse_zero(arraytype::Type{<:AbstractArray}, dims::Tuple{Vararg{Int}})
  a = arraytype(undef, dims)
  sparse_fill!(a, zero(eltype(a)))
  return a
end

function sparse_one!(a::AbstractMatrix)
  sparse_zero!(a)
  m, n = size(a)
  @assert m == n
  for i in 1:m
    a[i, i] = one(eltype(a))
  end
  return a
end

# TODO: Make `sparse_one!`?
function sparse_one(a::AbstractMatrix)
  a = sparse_zero(a)
  sparse_one!(a)
  return a
end

# TODO: Use `sparse_mapreduce(==, &, a1, a2)`?
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
  dest_cartesian_inds = CartesianIndices(dims)
  for I in stored_indices(a)
    a_reshaped[dest_cartesian_inds[linear_inds[I]]] = a[I]
  end
  return a_reshaped
end
