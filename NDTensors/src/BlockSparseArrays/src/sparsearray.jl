# TODO: Define a constructor with a default `zero`.
struct SparseArray{T,N,Zero} <: AbstractArray{T,N}
  data::Dictionary{CartesianIndex{N},T}
  dims::NTuple{N,Int}
  zero::Zero
end

Base.size(a::SparseArray) = a.dims

function Base.setindex!(a::SparseArray{T,N}, v, I::CartesianIndex{N}) where {T,N}
  set!(a.data, I, v)
  return a
end
function Base.setindex!(a::SparseArray{T,N}, v, I::Vararg{Int,N}) where {T,N}
  return setindex!(a, v, CartesianIndex(I))
end

function Base.getindex(a::SparseArray{T,N}, I::CartesianIndex{N}) where {T,N}
  # Don't evaluate `a.zero` unless it is needed.
  if !isassigned(a.data, I)
    return a.zero(T, I)
  end
  return a.data[I]
end
function Base.getindex(a::SparseArray{T,N}, I::Vararg{Int,N}) where {T,N}
  return getindex(a, CartesianIndex(I))
end

# SparseArrayKit.jl syntax
nonzero_keys(a::SparseArray) = keys(a.data)

# TODO: Make `PermutedSparseArray`.
# TODO: Distinguish between preserving structure or not.
# TODO: Make `similar`, either empty or preserving structure.
function Base.zero(a::PermutedDimsArray{<:Any,<:Any,<:Any,<:Any,<:SparseArray})
  # TODO: Make a simpler empty constructor.
  # SparseArray(size(a), parent(a).zero)
  # TODO: Define `zero_elt`.
  return SparseArray(
    Dictionary{CartesianIndex{ndims(a)},eltype(a)}(), size(a), parent(a).zero
  )
end

# TODO: Distinguish between preserving structure or not.
# TODO: Make `similar`, either empty or preserving structure.
# TODO: Combine with definition above.
function Base.zero(a::SparseArray)
  # TODO: Make a simpler empty constructor.
  # SparseArray(size(a), parent(a).zero)
  # TODO: Define `zero_elt`.
  return SparseArray(
    Dictionary{CartesianIndex{ndims(a)},eltype(a)}(), size(a), a.zero
  )
end

## # TODO: Make `PermutedSparseArray`.
## function map_nonzeros!(
##   f, a_dest::AbstractArray, a_src::Union{SparseArray,PermutedDimsArray{<:Any,<:Any,<:Any,<:Any,<:SparseArray}}
## )
##   for index in nonzero_keys(a_src)
##     a_dest[index] = f(a_src[index])
##   end
##   return a_dest
## end
## 
## # TODO: Make `PermutedSparseArray`.
## function map_nonzeros(f, a::Union{SparseArray,PermutedDimsArray{<:Any,<:Any,<:Any,<:Any,<:SparseArray}})
##   a_dest = zero(a)
##   map_nonzeros!(f, a_dest, a)
##   return a_dest
## end
## 
## function Base.map(f, a::PermutedDimsArray{<:Any,<:Any,<:Any,<:Any,<:SparseArray})
##   return map_nonzeros(f, a)
## end

const SparseArrayLike{T,N,Zero} = Union{SparseArray{T,N,Zero},PermutedDimsArray{T,N,<:Any,<:Any,SparseArray{T,N,Zero}}}

# TODO: Include `PermutedDimsArray`.
# TODO: Define `SparseArrayLike`.
function map_nonzeros!(f, a_dest::AbstractArray, as::SparseArrayLike...)
  @assert allequal(axes.(as))
  # TODO: Define `nonzero_keys` for multiple arrays, maybe with union?
  for index in union(nonzero_keys.(as)...)
    a_dest[index] = f(map(a -> a[index], as)...)
  end
  return a_dest
end

function map_nonzeros(f, as::SparseArrayLike...)
  @assert allequal(axes.(as))
  a_dest = zero(first(as))
  map!(f, a_dest, as...)
  return a_dest
end

function Base.map!(f, a_dest::AbstractArray, as::SparseArrayLike...)
  # TODO: Check `f` preserves zero.
  map_nonzeros!(f, a_dest, as...)
  return a_dest
end

function Base.map(f, as::SparseArrayLike...)
  # TODO: Check `f` preserves zero.
  return map_nonzeros(f, as...)
end
