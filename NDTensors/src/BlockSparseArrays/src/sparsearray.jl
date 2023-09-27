struct SparseArray{T,N} <: AbstractArray{T,N}
  data::Dictionary{CartesianIndex{N},T}
  dims::NTuple{N,Int64}
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
  return get(a.data, I, nothing)
end
function Base.getindex(a::SparseArray{T,N}, I::Vararg{Int,N}) where {T,N}
  return getindex(a, CartesianIndex(I))
end

# `getindex` but uses a default if the value is
# structurally zero.
function get_nonzero(a::SparseArray{T,N}, I::CartesianIndex{N}, zero) where {T,N}
  @boundscheck checkbounds(a, I)
  return get(a.data, I, zero)
end
function get_nonzero(a::SparseArray{T,N}, I::NTuple{N,Int}, zero) where {T,N}
  return get_nonzero(a, CartesianIndex(I), zero)
end
