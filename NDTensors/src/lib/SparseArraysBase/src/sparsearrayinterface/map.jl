using Base.Broadcast: BroadcastStyle, combine_styles
using Compat: allequal
using LinearAlgebra: LinearAlgebra

# Represents a value that isn't stored
# Used to hijack dispatch
struct NotStoredValue{Value}
  value::Value
end
value(v::NotStoredValue) = v.value
stored_length(::NotStoredValue) = false
Base.:*(x::Number, y::NotStoredValue) = false
Base.:*(x::NotStoredValue, y::Number) = false
Base.:/(x::NotStoredValue, y::Number) = false
Base.:+(::NotStoredValue, ::NotStoredValue...) = false
Base.:-(::NotStoredValue, ::NotStoredValue...) = false
Base.:+(x::Number, ::NotStoredValue...) = x
Base.iszero(::NotStoredValue) = true
Base.isreal(::NotStoredValue) = true
Base.conj(x::NotStoredValue) = conj(value(x))
Base.iterate(x::NotStoredValue) = (x, nothing)
Base.mapreduce(f, op, x::NotStoredValue) = f(x)
Base.zero(x::NotStoredValue) = zero(value(x))
LinearAlgebra.norm(x::NotStoredValue, p::Real=2) = zero(value(x))

notstored_index(a::AbstractArray) = NotStoredIndex(first(eachindex(a)))

# Get some not-stored value
function get_notstored(a::AbstractArray)
  return sparse_getindex(a, notstored_index(a))
end

function apply_notstored(f, as::Vararg{AbstractArray})
  return apply(f, NotStoredValue.(get_notstored.(as))...)
end

function apply(f, xs::Vararg{NotStoredValue})
  return f(xs...)
  #return try
  #  return f(xs...)
  #catch
  #  f(value(x))
  #end
end

# Test if the function preserves zero values and therefore
# preserves the sparsity structure.
function preserves_zero(f, as...)
  # return iszero(f(map(get_notstored, as)...))
  return iszero(apply_notstored(f, as...))
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
  # Need to zero out the destination.
  sparse_zero!(a_dest)
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
  return sparse_map!(combine_styles(as...), f, a_dest, as...)
end

function sparse_map!(::BroadcastStyle, f, a_dest::AbstractArray, as::AbstractArray...)
  @assert allequal(axes.((a_dest, as...)))
  if preserves_zero(f, as...)
    # Remove aliases to avoid overwriting inputs.
    as = map(a -> Base.unalias(a_dest, a), as)
    sparse_map_stored!(f, a_dest, as...)
  else
    sparse_map_all!(f, a_dest, as...)
  end
  return a_dest
end

# `f::typeof(norm)`, `op::typeof(max)` used by `norm`.
function reduce_init(f, op, a)
  return f(zero(eltype(a)))
end

# TODO: Generalize to multiple arguements.
# TODO: Define `sparse_mapreducedim!`.
function sparse_mapreduce(f, op, a::AbstractArray; init=reduce_init(f, op, a), kwargs...)
  output = mapreduce(f, op, sparse_storage(a); init, kwargs...)
  f_notstored = apply_notstored(f, a)
  @assert isequal(op(output, eltype(output)(f_notstored)), output)
  return output
end
