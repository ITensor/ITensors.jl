# This is used when a sparse output structure not matching
# the input structure is needed, for example when reshaping
# a DiagonalArray. Overload:
#
# sparse_similar(a::AbstractArray, elt::Type, dims::Tuple{Vararg{Int}})
#
# as needed.
function sparse_similar(a::AbstractArray, elt::Type)
  return similar(a, elt, size(a))
end

function sparse_similar(a::AbstractArray, dims::Tuple{Vararg{Int}})
  return sparse_similar(a, eltype(a), dims)
end

function sparse_similar(a::AbstractArray)
  return sparse_similar(a, eltype(a), size(a))
end

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
    # This checks that `x` is compatible
    # with `eltype(a)`.
    x = convert(eltype(a), x)
    sparse_zero!(a)
    return a
  end
  for I in eachindex(a)
    a[I] = x
  end
  return a
end

# This could just call `sparse_fill!`
# but it avoids a zero construction and check.
function sparse_zero!(a::AbstractArray)
  dropall!(a)
  sparse_zerovector!(a)
  return a
end

function sparse_zero(a::AbstractArray)
  # Need to overload `similar` for custom types
  a = similar(a)
  sparse_zerovector!(a)
  return a
end

# TODO: Is this a good definition?
function sparse_zero(arraytype::Type{<:AbstractArray}, dims::Tuple{Vararg{Int}})
  a = arraytype(undef, dims)
  sparse_zerovector!(a)
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

function sparse_one(a::AbstractMatrix)
  a = sparse_zero(a)
  sparse_one!(a)
  return a
end

# TODO: Use `sparse_mapreduce(==, &, a1, a2)`?
function sparse_isequal(a1::AbstractArray, a2::AbstractArray)
  Is = collect(stored_indices(a1))
  intersect!(Is, stored_indices(a2))
  if !(length(Is) == stored_length(a1) == stored_length(a2))
    return false
  end
  for I in Is
    a1[I] == a2[I] || return false
  end
  return true
end

function sparse_reshape!(a_dest::AbstractArray, a_src::AbstractArray, dims)
  @assert length(a_src) == prod(dims)
  sparse_zero!(a_dest)
  linear_inds = LinearIndices(a_src)
  dest_cartesian_inds = CartesianIndices(dims)
  for I in stored_indices(a_src)
    a_dest[dest_cartesian_inds[linear_inds[I]]] = a_src[I]
  end
  return a_dest
end

function sparse_reshape(a::AbstractArray, dims)
  a_reshaped = sparse_similar(a, dims)
  sparse_reshape!(a_reshaped, a, dims)
  return a_reshaped
end
