using Base.Broadcast: BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle, Broadcasted
using ..BroadcastMapConversion: map_function, map_args

struct SparseArrayDOKStyle{N} <: AbstractArrayStyle{N} end

function Broadcast.BroadcastStyle(::Type{<:SparseArrayDOK{<:Any,N}}) where {N}
  return SparseArrayDOKStyle{N}()
end

SparseArrayDOKStyle(::Val{N}) where {N} = SparseArrayDOKStyle{N}()
SparseArrayDOKStyle{M}(::Val{N}) where {M,N} = SparseArrayDOKStyle{N}()

Broadcast.BroadcastStyle(a::SparseArrayDOKStyle, ::DefaultArrayStyle{0}) = a
function Broadcast.BroadcastStyle(::SparseArrayDOKStyle{N}, a::DefaultArrayStyle) where {N}
  return BroadcastStyle(DefaultArrayStyle{N}(), a)
end
function Broadcast.BroadcastStyle(
  ::SparseArrayDOKStyle{N}, ::Broadcast.Style{Tuple}
) where {N}
  return DefaultArrayStyle{N}()
end

# TODO: Use `allocate_output`, share logic with `map`.
function Base.similar(bc::Broadcasted{<:SparseArrayDOKStyle}, elt::Type)
  # TODO: Is this a good definition? Probably should check that
  # they have consistent axes.
  return similar(first(map_args(bc)), elt)
end

# Broadcasting implementation
function Base.copyto!(
  dest::SparseArrayDOK{<:Any,N}, bc::Broadcasted{SparseArrayDOKStyle{N}}
) where {N}
  # convert to map
  # flatten and only keep the AbstractArray arguments
  map!(map_function(bc), dest, map_args(bc)...)
  return dest
end
