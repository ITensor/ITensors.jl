using Base.Broadcast: BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle, Broadcasted
using ..BroadcastMapConversion: map_function, map_args

struct BlockSparseArrayStyle{N} <: AbstractArrayStyle{N} end

# Define for new sparse array types.
# function Broadcast.BroadcastStyle(arraytype::Type{<:MyBlockSparseArray})
#   return BlockSparseArrayStyle{ndims(arraytype)}()
# end

BlockSparseArrayStyle(::Val{N}) where {N} = BlockSparseArrayStyle{N}()
BlockSparseArrayStyle{M}(::Val{N}) where {M,N} = BlockSparseArrayStyle{N}()

Broadcast.BroadcastStyle(a::BlockSparseArrayStyle, ::DefaultArrayStyle{0}) = a
function Broadcast.BroadcastStyle(::BlockSparseArrayStyle{N}, a::DefaultArrayStyle) where {N}
  return BroadcastStyle(DefaultArrayStyle{N}(), a)
end
function Broadcast.BroadcastStyle(::BlockSparseArrayStyle{N}, ::Broadcast.Style{Tuple}) where {N}
  return DefaultArrayStyle{N}()
end

# TODO: Use `allocate_output`, share logic with `map`.
function Base.similar(bc::Broadcasted{<:BlockSparseArrayStyle}, elt::Type)
  # TODO: Is this a good definition? Probably should check that
  # they have consistent axes.
  return similar(first(map_args(bc)), elt)
end

# Broadcasting implementation
function Base.copyto!(
  dest::AbstractArray{<:Any,N}, bc::Broadcasted{BlockSparseArrayStyle{N}}
) where {N}
  # convert to map
  # flatten and only keep the AbstractArray arguments
  blocksparse_map!(map_function(bc), dest, map_args(bc)...)
  return dest
end
