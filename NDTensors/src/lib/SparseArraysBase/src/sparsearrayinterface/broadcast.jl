using Base.Broadcast: BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle, Broadcasted
using ..BroadcastMapConversion: map_function, map_args

struct SparseArrayStyle{N} <: AbstractArrayStyle{N} end

# Define for new sparse array types.
# function Broadcast.BroadcastStyle(arraytype::Type{<:MySparseArray})
#   return SparseArrayStyle{ndims(arraytype)}()
# end

SparseArrayStyle(::Val{N}) where {N} = SparseArrayStyle{N}()
SparseArrayStyle{M}(::Val{N}) where {M,N} = SparseArrayStyle{N}()

Broadcast.BroadcastStyle(a::SparseArrayStyle, ::DefaultArrayStyle{0}) = a
function Broadcast.BroadcastStyle(::SparseArrayStyle{N}, a::DefaultArrayStyle) where {N}
  return BroadcastStyle(DefaultArrayStyle{N}(), a)
end
function Broadcast.BroadcastStyle(::SparseArrayStyle{N}, ::Broadcast.Style{Tuple}) where {N}
  return DefaultArrayStyle{N}()
end

# TODO: Use `allocate_output`, share logic with `map`.
function Base.similar(bc::Broadcasted{<:SparseArrayStyle}, elt::Type)
  # TODO: Is this a good definition? Probably should check that
  # they have consistent axes.
  return similar(first(map_args(bc)), elt)
end

# Broadcasting implementation
function Base.copyto!(
  dest::AbstractArray{<:Any,N}, bc::Broadcasted{SparseArrayStyle{N}}
) where {N}
  # convert to map
  # flatten and only keep the AbstractArray arguments
  sparse_map!(map_function(bc), dest, map_args(bc)...)
  return dest
end
