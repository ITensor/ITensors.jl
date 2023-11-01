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
function Base.zero(a::PermutedDimsArray{<:Any,<:Any,<:Any,<:Any,<:SparseArray})
  # TODO: Make a simpler empty constructor.
  # SparseArray(size(a), parent(a).zero)
  # TODO: Define `zero_elt`.
  return SparseArray(
    Dictionary{CartesianIndex{ndims(a)},eltype(a)}(), size(a), parent(a).zero
  )
end

# TODO: Make `PermutedSparseArray`.
function map_nonzeros!(
  f, a_dest::AbstractArray, a_src::PermutedDimsArray{<:Any,<:Any,<:Any,<:Any,<:SparseArray}
)
  for index in nonzero_keys(a_src)
    a_dest[index] = f(a_src[index])
  end
  return a_dest
end

# TODO: Make `PermutedSparseArray`.
function map_nonzeros(f, a::PermutedDimsArray{<:Any,<:Any,<:Any,<:Any,<:SparseArray})
  a_dest = zero(a)
  map_nonzeros!(f, a_dest, a)
  return a_dest
end
