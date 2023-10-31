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
  return get(a.data, I, a.zero(T, I))
end
function Base.getindex(a::SparseArray{T,N}, I::Vararg{Int,N}) where {T,N}
  return getindex(a, CartesianIndex(I))
end
