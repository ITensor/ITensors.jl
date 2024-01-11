using FillArrays: broadcasted_fill, broadcasted_zeros, getindex_value

abstract type ZeroPreserving end
struct IsZeroPreserving <: ZeroPreserving end
struct NotZeroPreserving <: ZeroPreserving end

# Assume operations don't preserve zeros for safety
ZeroPreserving(x) = NotZeroPreserving()
ZeroPreserving(::typeof(Complex)) = IsZeroPreserving()
ZeroPreserving(::typeof(Real)) = IsZeroPreserving()

function Broadcast.broadcasted(style::Broadcast.DefaultArrayStyle, f, a::UnallocatedZeros)
  return _broadcasted(style, f, ZeroPreserving(f), a)
end

function _broadcasted(
  style::Broadcast.DefaultArrayStyle, f, ::IsZeroPreserving, a::UnallocatedZeros
)
  z = f.(parent(a))
  return broadcasted_zeros(f, a, eltype(z), axes(z))
end

function _broadcasted(
  style::Broadcast.DefaultArrayStyle, f, ::NotZeroPreserving, a::UnallocatedZeros
)
  z = f.(parent(a))
  return broadcasted_fill(f, a, getindex_value(z), axes(z))
end
