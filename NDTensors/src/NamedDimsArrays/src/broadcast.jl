using Base.Broadcast: BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle, Broadcasted

struct NamedDimsArrayStyle{N} <: AbstractArrayStyle{N} end

function Broadcast.BroadcastStyle(::Type{<:AbstractNamedDimsArray{<:Any,N}}) where {N}
  return NamedDimsArrayStyle{N}()
end

NamedDimsArrayStyle(::Val{N}) where {N} = NamedDimsArrayStyle{N}()
NamedDimsArrayStyle{M}(::Val{N}) where {M,N} = NamedDimsArrayStyle{N}()

Broadcast.BroadcastStyle(a::NamedDimsArrayStyle, ::DefaultArrayStyle{0}) = a
function Broadcast.BroadcastStyle(::NamedDimsArrayStyle{N}, a::DefaultArrayStyle) where {N}
  return BroadcastStyle(DefaultArrayStyle{N}(), a)
end
function Broadcast.BroadcastStyle(
  ::NamedDimsArrayStyle{N}, ::Broadcast.Style{Tuple}
) where {N}
  return DefaultArrayStyle{N}()
end

# TODO: Is this needed?
# Define `output_names`, like `allocate_output`.
# function dimnames(bc::Broadcasted{<:NamedDimsArrayStyle})
#   return dimnames(first(map_args(bc)))
# end

# TODO: Use `allocate_output`, share logic with `map`.
function Base.similar(bc::Broadcasted{<:NamedDimsArrayStyle}, elt::Type)
  return similar(first(map_args(bc)), elt)
end

# Broadcasting implementation
function Base.copyto!(
  dest::AbstractNamedDimsArray{<:Any,N}, bc::Broadcasted{NamedDimsArrayStyle{N}}
) where {N}
  # convert to map
  # flatten and only keep the AbstractArray arguments
  map!(map_function(bc), dest, map_args(bc)...)
  return dest
end
