using Dictionaries: Dictionary, set!

# TODO: Define a constructor with a default `zero`.
struct SparseArrayDOK{T,N,Zero} <: AbstractArray{T,N}
  data::Dictionary{CartesianIndex{N},T}
  dims::NTuple{N,Int}
  zero::Zero
end

# Constructors
function SparseArrayDOK{T,N,Zero}(dims::Tuple{Vararg{Int}}, zero) where {T,N,Zero}
  return SparseArrayDOK{T,N,Zero}(default_data(T, N), dims, zero)
end

function SparseArrayDOK{T,N}(dims::Tuple{Vararg{Int}}, zero) where {T,N}
  return SparseArrayDOK{T,N,typeof(zero)}(dims, zero)
end

function SparseArrayDOK{T,N}(dims::Tuple{Vararg{Int}}) where {T,N}
  return SparseArrayDOK{T,N}(dims, default_zero())
end

function SparseArrayDOK{T}(dims::Tuple{Vararg{Int}}) where {T}
  return SparseArrayDOK{T,length(dims)}(dims)
end

function SparseArrayDOK{T}(dims::Int...) where {T}
  return SparseArrayDOK{T}(dims)
end

# Specify zero function
function SparseArrayDOK{T}(dims::Tuple{Vararg{Int}}, zero) where {T}
  return SparseArrayDOK{T,length(dims)}(dims, zero)
end

# undef
function SparseArrayDOK{T,N,Zero}(
  ::UndefInitializer, dims::Tuple{Vararg{Int}}, zero
) where {T,N,Zero}
  return SparseArrayDOK{T,N,Zero}(dims, zero)
end

function SparseArrayDOK{T,N}(::UndefInitializer, dims::Tuple{Vararg{Int}}, zero) where {T,N}
  return SparseArrayDOK{T,N}(dims, zero)
end

function SparseArrayDOK{T,N}(::UndefInitializer, dims::Tuple{Vararg{Int}}) where {T,N}
  return SparseArrayDOK{T,N}(dims)
end

function SparseArrayDOK{T}(::UndefInitializer, dims::Tuple{Vararg{Int}}) where {T}
  return SparseArrayDOK{T}(dims)
end

function SparseArrayDOK{T}(::UndefInitializer, dims::Int...) where {T}
  return SparseArrayDOK{T}(dims...)
end

function SparseArrayDOK{T}(::UndefInitializer, dims::Tuple{Vararg{Int}}, zero) where {T}
  return SparseArrayDOK{T}(dims, zero)
end

# Required `SparseArrayInterface` interface
nonzero_structure(a::SparseArrayDOK) = a.data
# TODO: Make this a generic function.
nonzero_keys(a::SparseArrayDOK) = keys(nonzero_structure(a))

# Optional `SparseArrayInterface` interface
# TODO: Use `SetParameters`.
zero_value(a::SparseArrayDOK, I) = a.zero(eltype(a), I)

# Accessors
Base.size(a::SparseArrayDOK) = a.dims

function Base.getindex(a::SparseArrayDOK{<:Any,N}, I::Vararg{Int,N}) where {N}
  return a[CartesianIndex(I)]
end

# TODO: Rename `getindex_sparse` or `sparse_getindex`.
function Base.getindex(a::SparseArrayDOK{<:Any,N}, I::CartesianIndex{N}) where {N}
  if !is_structural_nonzero(a, I)
    return getindex_zero(a, I)
  end
  return getindex_nonzero(a, I)
end

# `SparseArrayInterface` interface
function setindex_zero!(a::SparseArrayDOK, value, I)
  # TODO: This is specific to the `Dictionaries.jl`
  # interface, make more generic?
  set!(nonzero_structure(a), I, value)
  return a
end

function Base.setindex!(a::SparseArrayDOK{<:Any,N}, value, I::Vararg{Int,N}) where {N}
  a[CartesianIndex(I)] = value
  return a
end

function Base.setindex!(a::SparseArrayDOK{<:Any,N}, value, I::CartesianIndex{N}) where {N}
  if !is_structural_nonzero(a, I)
    setindex_zero!(a, value, I)
  end
  setindex_nonzero!(a, value, I)
  return a
end

# similar
function Base.similar(a::SparseArrayDOK)
  return similar(a, eltype(a))
end

function Base.similar(a::SparseArrayDOK, elt::Type)
  return similar(a, elt, size(a))
end

function Base.similar(a::SparseArrayDOK, elt::Type, dims::Tuple{Vararg{Int}})
  # TODO: Accessor function for `a.zero`.
  return SparseArrayDOK{elt}(undef, dims, a.zero)
end
